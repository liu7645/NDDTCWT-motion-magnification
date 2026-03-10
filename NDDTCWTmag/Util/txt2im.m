function txtIm = txt2im(txt, bgColor, txtColor, txtWidth, varargin)

if nargin<2 | isempty(bgColor)
    bgColor = [0,0,0];
end
if nargin<3 | isempty(txtColor)
    txtColor = [1,1,1];
end
    
h = figure('Visible', 'Off');
set(h, 'Color', bgColor);
set(gca,'Position',[0 0 1 1]);
axis tight; axis off; axis ij;
axis([0 1 0 1]);

% render text
t = text(.5, .5, txt, ...
         'VerticalAlignment', 'Middle', 'HorizontalAlignment', 'center', ...
         'Color', txtColor, ...
         varargin{:});

ax = axis;
ext = get(t, 'Extent');
pos = get(h,'Position');  
% set(h, 'Position', [50 50 (pos(3) * ext(3)/(ax(2)-ax(1))) (pos(4) * ext(4)/(ax(4)-ax(3)))]);
set(h, 'Position', [pos(1) pos(2) txtWidth (pos(4) * ext(4)/(ax(4)-ax(3)))]);

if pos(3)*ext(3)/(ax(2)-ax(1)) > txtWidth
    warning(sprintf('Text txtWidth is larger than %d',txtWidth));
end

drawnow;

TtxtIm = getFrameWithoutFocus(h);
txtIm = TtxtIm;
%T = getframe(h);
%txtIm = T.cdata;


tmp = size(txtIm,2)-txtWidth;
if tmp>0
    left = round(tmp/2);
    right = tmp-left;
    txtIm = TtxtIm(:,left+1:end-right,:);
end
out= zeros(size(txtIm,1),txtWidth,size(txtIm,3));
out(:,1:size(txtIm,2),:) = txtIm;
txtIm = out;
close(h);
pause(0.1);

% [txtImHeight,txtImWidth,tmp]= size(txtIm);

% if txtWidth > txtImWidth
%     leftPadSize = floor((txtWidth-txtImWidth)/2);
%     rightPadSize = txtWidth-leftPadSize-txtImWidth;
%     for i=1:3
%     txtIm2(:,:,i) = [ones(txtImHeight,leftPadSize)*bgColor(i),...
%                      txtIm(:,:,i),...
%                      ones(txtImHeight,rightPadSize)*bgColor(i)];
%     end
%     txtIm = txtIm2;
% end
