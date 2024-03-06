function varargout = histnorm(varargin)
% Function to compute normalized histogram.
% Input:
%   varargin: Variable input arguments. Can contain data to be histogrammed            
% Output:
%   varargout: Variable output arguments. Contains the normalized histogram
%              and bin edges.
% Note: If no output is requested, the function automatically plots the
%       normalized histogram.

% Flag to determine if plotting is requested, initialized to 0
doPlot = 0; 
if ischar (varargin{end}) && strcmpi ('plot', varargin{end})
    doPlot = 1;
    varargin = varargin(1:end-1);
elseif nargout == 0
    doPlot = 1;    
end

% Compute the histogram and store the counts and bin edges
h = histogram (varargin{:});
xo = h.Values;
no = h.BinEdges + 0.5*h.BinWidth;
no = no(1:length(no)-1);
binwidths = h.BinWidth;

% Normalize the histogram so that the area under the histogram is 1
xonorm = xo/sum (xo .* binwidths);

% Prepare output arguments
varargout = {xonorm, no};
varargout = varargout(1:nargout);

% Plot the histogram if doPlot is true
if doPlot
    cax = axescheck(varargin{:});
    hist (varargin{:});
    if isempty (cax)
        cax = gca;
    end
    ch = findobj (get (cax, 'children'), 'type', 'patch'); ch = ch(1);
    vertices = get (ch, 'vertices');
    for idx = 1:numel (xonorm)
        vertices((idx-1)*5+[3 4],2) = xonorm(idx);     % hope it works :)
    end
    set (ch, 'vertices', vertices);
end
