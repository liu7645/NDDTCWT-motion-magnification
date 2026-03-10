function [ colored_image ] = ApplyColorMap( image, colormap )
    % Takes a grayscale image and maps each grayscale value to a color
    % specified in colormap to output colored_image.
    %
    % Neal Wadhwa, MIT 2016
    colored_image = zeros(size(image,1), size(image, 2), 3);
    number_of_colors = size(colormap, 1);
    intensity_range = linspace(0, 1, number_of_colors);
    for c  = 1:3
        colored_image(:,:,c) = interp1(intensity_range, colormap(:,c), image,'linear', 'extrap');
    end
    colored_image = clip(colored_image, 0, 1);
end

