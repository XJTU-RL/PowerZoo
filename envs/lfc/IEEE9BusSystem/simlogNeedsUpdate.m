function output = simlogNeedsUpdate(simlog, modelName)
% Function to check if simlog needs to be updated. This is an
% undocumented function used only by example models to determine the
% need for resimulation. Use of this function for other
% purposes is not recommended

% Copyright 2019-2024 The MathWorks, Inc.

time_stamp = simscape.logging.timestamp(simlog);
if get_param(modelName,'RTWModifiedTimeStamp') == double(time_stamp)
    output = false;
else
    output = true;
end
end
