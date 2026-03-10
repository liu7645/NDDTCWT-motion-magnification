function delta = accelerationFilter( delta,sigma, fh )
    len = size(delta,3);

    
    kernel(1,1,:) = [-1 2 -1];

    N = ceil(3*sigma);
    if (sigma ~=0)
        gaussKernel = exp(-(-N:N).^2./(2*sigma.^2));
        gaussKernel = gaussKernel./sum(gaussKernel);
        gaussKernelT(1,1,:) = gaussKernel;
    else
        gaussKernelT(1,1,1) = 1;
    end
        
    M = size(delta,1);
    batches = 40;    
    batchSize = ceil(M/batches);    
    init = zeros(size(delta(:,:,2),1), size(delta(:,:,2),2), 2);
    for k = 1:batches
        idx = 1+batchSize*(k-1):min(k*batchSize, M);
        temp = convn(delta(idx,:,:), kernel, 'valid');
        temp1 = zeros(size(delta(idx,:,:)));
        temp1(:,:,2:end-1) = temp;
        temp1 = convn(temp1, gaussKernelT, 'same');
        if (not(isempty(temp1)))
            delta(idx,:,:) = temp1;
        end
    end

end

