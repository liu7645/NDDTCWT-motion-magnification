function varargout = RunOnceOnly( target_mat_filename, func, runAnyway)
    % Runs a time-consuming operation and saves the results. Subsequent
    % runs of function will just load the results, saving time.
    % If target_mat_filename doens't exist, run func and save output to
    % target_mat_filename. If it exists, load target_mat_filename and
    % return the function's output.
    % If runAnyway is true, runs function anyway.
    %
    % Neal Wadhwa, MIT 2016
    if (nargin < 3)
        runAnyway = false;
    end
    numOutputs = nargout;
    if (or(~exist(target_mat_filename, 'file'), runAnyway))
       cmd_str = '[varargout{1}, ';
       for k = 2:numOutputs
          cmd_str = [cmd_str sprintf('varargout{%d}, ', k)]; 
       end
       cmd_str = [cmd_str ' ] = func();'];        
       eval(cmd_str);
       save(target_mat_filename, 'varargout', '-v7.3');
    else
       fprintf('%s already exists! Not recomputing.\n', target_mat_filename); 
       F = load(target_mat_filename);
       varargout = F.varargout;
    end

end

