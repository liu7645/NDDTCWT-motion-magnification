// The bulk of the motion computation code is in this file. It includes 
// computing the spatial gradients of the phase and performing
// weighted least squares to estimate motion at every pixel.
#include <Halide.h>

const float pi = 3.1415925638f;
const float eps = 1e-6f;
using namespace Halide;
using namespace Halide::BoundaryConditions;

Var x, y, c;
Expr space_blur_extent;


// kernelx and kernely are expected to be precomputed of length 5
// insertion_point and insertion_var are used for scheduling
Func Differentiate(Func input, Func kernelx, Func kernely, Func insertion_point, Var insertion_var) {
    // Convolve input with separable kernel
    RDom rx(-2, 5);
    Func blurx("blurx");
    Func blury("blury");
    blurx(x, y, c) = sum(input(x + rx, y, c) * kernelx(rx));
    blury(x, y, c) = sum(blurx(x, y+rx, c) * kernely(rx));
    
    // Schedule for differentiation
    Var xi, yi;
    blury.split(y, y, yi, 32)
        .compute_at(insertion_point, insertion_var)    
        .vectorize(x, 8);
    blurx.store_at(blury, y)
        .compute_at(blury, yi)
        .vectorize(x, 8);
    return blury;
}


// Converts tuning_freq from integer coordinates to normalized frequency
// See adjustTuningFrequency.m for more details.
Func AdjustTuningFrequency(Func tuning_freq, Expr width, Expr height) {
    Func adjusted_tuning_freq("adjusted_tuning_freq");
    adjusted_tuning_freq(x, y) = undef<float>();
    adjusted_tuning_freq(x, 0) = 2*pi*((tuning_freq(x ,0)-1)/width);
    adjusted_tuning_freq(x, 1) = 2*pi*((tuning_freq(x, 1)-1)/height);
    adjusted_tuning_freq(x, y) = select(adjusted_tuning_freq(x, y) > pi,
                                        adjusted_tuning_freq(x, y) - 2*pi,
                                        adjusted_tuning_freq(x, y));
    adjusted_tuning_freq(x, y) = select(adjusted_tuning_freq(x, y) < - pi,
                                        adjusted_tuning_freq(x, y) + 2*pi,
                                        adjusted_tuning_freq(x, y));
    
    adjusted_tuning_freq.compute_root();   
    return adjusted_tuning_freq;
}


// Computes
Func blur_and_sum(Func input1, Func input2, Func kernel_1D, Expr levels, Func insertion_point, Var insertion_var) {
    RDom rx(-space_blur_extent, 2*space_blur_extent + 1);        
    RDom rc(0, levels);  
    Func product("X11_product");
    Func blurx("X11_blurx");
    Func blury("X11_blury");
    product(x, y, c) = input1(x, y, c) * input2(x, y, c);
    blurx(x, y, c) = sum(product(x + rx, y, c) * kernel_1D(rx));
    blury(x, y, c) = sum(blurx(x, y+ rx, c) * kernel_1D(rx));               
    Func output;
    output(x, y) = sum(blury(x, y, rc)); 
    // Schedule    
    Var xi, yi;
    output.compute_at(insertion_point, insertion_var)
        .vectorize(x, 8);
    blury.split(y,y,yi,32)
         .compute_at(insertion_point, insertion_var)
         .vectorize(x,8);
    blurx.store_at(blury, y)
         .compute_at(blury, yi)
         .vectorize(x, 8);  
    return output;
}


// Generator for computing motion
class ComputeMotionHalideMex : public Generator<ComputeMotionHalideMex> {
public:
    // Data coming in from matlab
    ImageParam phit{Float(32), 3, "phit"};  // Phase differences
    ImageParam response_real{Float(32), 3, "response_real"};
    ImageParam response_imag{Float(32), 3, "response_imag"};
    ImageParam mod_real_I{Float(32), 3, "mod_real"}; // Real part of modulation
    ImageParam mod_imag_I{Float(32), 3, "mod_imag"};
    ImageParam tuning_freq{Float(32), 2, "tuning_freq"}; 
    Param<float> space_blur{"space_blur"}; // Space_blur parameter in pixels
  
        

    Func build() {

        Func final_output("final_output");
        Var xi, yi, f;
        
        // We get transpose from MATLAB
        Expr width = phit.height();
        Expr height = phit.width();
        Expr levels = phit.channels();
        
        // Creates spatial blur kernel
        space_blur_extent = 2*ceil(space_blur);
        RDom rx(-space_blur_extent, 2*space_blur_extent + 1);        
        Func kernel_1D("kernel_1D");
        Func kernel_2;
        kernel_2(x) = exp(-(x*x)/(2*space_blur*space_blur+eps));
        Expr normalization = sum(kernel_2(rx));
        kernel_1D(x) = exp(-(x*x)/(2*space_blur*space_blur+eps))/normalization;        
        kernel_1D.compute_root();
        
        // Adds repeating boundary conditions to edges, can cause 
        // motion to be underestimated near the edge.
        Func phit_edge;
        Func weight_sq("weight_sq");
        phit_edge = repeat_edge(phit);                      
        Func response_real_edge = repeat_edge(response_real);
        Func response_imag_edge = repeat_edge(response_imag);
        Expr eps = 1e-16f;
        weight_sq(x, y, c) = response_real_edge(x, y, c)*response_real_edge(x, y, c) +
                             response_imag_edge(x, y, c)*response_imag_edge(x, y, c) + eps;        
        Func mod_real, mod_imag;
        mod_real = repeat_edge(mod_real_I);
        mod_imag = repeat_edge(mod_imag_I);
        
        // Adjusts tuning frequency, so modulation is computed correctly
        // See adjustTuningFrequency.m for more details
        Func tuning_freq_ff;
        tuning_freq_ff(x, y) = tuning_freq(x, y);
        Func adjusted_tuning_freq = AdjustTuningFrequency(tuning_freq_ff, width, height);
        
        // Precompuete derivative filter kernels. Uses five tap kernels
        // specified in Simoncelli, "DESIGN OF MULTI-DIMENSIONAL DERIVATIVE 
        // FILTERS" 1994
        Func deriv_kernel("deriv_kernel");
        deriv_kernel(x) = undef<float>();
        deriv_kernel(-2) = -0.109604f;
        deriv_kernel(-1) = -0.276691f;
        deriv_kernel(0) = 0.0f;
        deriv_kernel(1) = 0.276691f;
        deriv_kernel(2) = 0.109604f;
        deriv_kernel.compute_root();
        
        Func deriv_prefilter("deriv_prefilter");
        deriv_prefilter(x) = undef<float>();
        deriv_prefilter(-2) = 0.037659f;
        deriv_prefilter(-1) = 0.249153f;
        deriv_prefilter(0) = 0.426375f;
        deriv_prefilter(1) = 0.249153f;
        deriv_prefilter(2) = 0.037659f;
        deriv_prefilter.compute_root();
        
        // Compute spatial gradient of phase by taking the imaginary part
        // of the 
        Func phix, phiy;
        Func lowpass_real, lowpass_imag;
        lowpass_real(x, y, c) = response_real_edge(x, y, c)*mod_real(x, y, c) + response_imag_edge(x, y, c)*mod_imag(x, y, c);
        lowpass_imag(x, y, c) = response_imag_edge(x, y, c)*mod_real(x, y, c) - response_real_edge(x, y, c)*mod_imag(x, y, c);
        
        Func grady_real = Differentiate(lowpass_real, deriv_kernel, deriv_prefilter, final_output, f);        
        Func grady_imag = Differentiate(lowpass_imag, deriv_kernel, deriv_prefilter, final_output, f);
        Func gradx_real = Differentiate(lowpass_real, deriv_prefilter, deriv_kernel, final_output, f);
        Func gradx_imag = Differentiate(lowpass_imag, deriv_prefilter, deriv_kernel, final_output, f);
        Func gradx_real2, gradx_imag2, grady_real2, grady_imag2;
        gradx_real2(x, y, c) = gradx_real(x, y, c) * mod_real(x, y, c) - 
                              (gradx_imag(x, y, c) * mod_imag(x, y, c) +
                              adjusted_tuning_freq(c, 0) * response_imag_edge(x, y, c));
        
        gradx_imag2(x, y, c) = gradx_imag(x, y, c) * mod_real(x, y, c) + 
                              gradx_real(x, y, c) * mod_imag(x, y, c) +
                              adjusted_tuning_freq(c, 0) * response_real_edge(x, y, c);
        
        grady_real2(x, y, c) = grady_real(x, y, c) * mod_real(x, y, c) - 
                              (grady_imag(x, y, c) * mod_imag(x, y, c) +
                              adjusted_tuning_freq(c, 1) * response_imag_edge(x, y, c));
        
        grady_imag2(x, y, c) = grady_imag(x, y, c) * mod_real(x, y, c) + 
                              grady_real(x, y, c) * mod_imag(x, y, c) +
                              adjusted_tuning_freq(c, 1) * response_real_edge(x, y, c);
     
        // Computes quantities for weighted least squares
        Func phixW("phixW");
        Func phiyW("phiyW");        
        phixW(x, y, c) = (response_real_edge(x, y, c) * gradx_imag2(x, y, c) -
                          response_imag_edge(x, y, c) * gradx_real2(x, y, c));
        phiyW(x, y, c) = (response_real_edge(x, y, c) * grady_imag2(x, y, c) -
                          response_imag_edge(x, y, c) * grady_real2(x, y, c));
        
        
        
        Func output("output");

        Func inverse_weight_sq;
        inverse_weight_sq(x, y, c) = 1.0f/weight_sq(x, y, c);
               
        phix(x, y, c) =  inverse_weight_sq(x, y, c)*phixW(x, y, c);
        phiy(x, y, c) =  inverse_weight_sq(x, y, c)*phiyW(x, y, c);        
        
        
        // Compute components of X^TWX, note that X21 = X12 and doesn't 
        // need to be computed
        Func X11 = blur_and_sum(phixW, phix, kernel_1D, levels, final_output, f);        
        Func X12 = blur_and_sum(phixW, phiy, kernel_1D, levels, final_output, f);        
        Func X22 = blur_and_sum(phiyW, phiy, kernel_1D, levels, final_output, f);        
        // Compute components of X^TWY
        Func Y1 =  blur_and_sum(phixW, phit_edge, kernel_1D, levels, final_output, f);       
        Func Y2 =  blur_and_sum(phiyW, phit_edge, kernel_1D, levels, final_output, f);       
        
        
        Func D("D"); // Discriminant of X^TWX
        D(x, y) = 1.0f/(X11(x, y) *X22(x, y)-X12(x,y)*X12(x,y));
        
        // Solve for the motion (X^TWX)^{-1}X^TWY
        output(x, y, c) = undef<float>();
        output(x, y, 0) = (-X22(x, y) *Y1(x, y)+X12(x,y)*Y2(x, y))*D(x,y);
        output(x, y, 1) = (X12(x, y) *Y1(x, y)-X11(x,y)*Y2(x, y))*D(x,y);
        

        final_output(x, y, c) = output(x, y, c);
        
        // Scheduling     
        int natural_vector_width = 8; 
        
        // Vectorize and parallelize the computation on tiles
        final_output.compute_root()
            .specialize(width >= 128 && height>=128)
            .tile(x, y, xi, yi, 64, 64)
            .fuse(x, y, f)
            .reorder(xi, yi, c, f)
            .vectorize(xi, natural_vector_width)
            .parallel(f,8);
        final_output.compute_root()
            .tile(x, y, xi, yi, 32, 32)
            .fuse(x, y, f)
            .reorder(xi, yi, c, f)
            .vectorize(xi, natural_vector_width)
            .parallel(f,8);
        
        D.compute_at(final_output, yi)
         .vectorize(x, natural_vector_width);
        inverse_weight_sq.compute_at(final_output, f)
            .vectorize(x, natural_vector_width);
        
        phixW.compute_at(final_output, f)            
            .vectorize(x, natural_vector_width);
        phiyW.compute_at(final_output, f)
            .vectorize(x, natural_vector_width);

        lowpass_real.compute_at(final_output, f)
            .vectorize(x, natural_vector_width);
        lowpass_imag.compute_at(final_output, f)
            .vectorize(x, natural_vector_width);
        return final_output;
    }
};

auto computeMotionHalidemex = RegisterGenerator<ComputeMotionHalideMex>("ComputeMotionHalideMex");
