% Code to plot simulation results from IEEE9BusSystem
%% Plot Description:
%
% This plot shows the rotor speed and terminal voltage of Generator-1
% at the swing bus, Bus-1, and the rotor speed and terminal voltage of
% Generator-2 and Generator-3 at the PV buses, Bus-2 and Bus-3.

% Copyright 2022-2023 The MathWorks, Inc.

% Generate new simulation results if they don't exist or if they need to be updated
if ~exist('simlog_IEEE9BusSystem', 'var') || ...
        simlogNeedsUpdate(simlog_IEEE9BusSystem, 'IEEE9BusSystem') 
    sim('IEEE9BusSystem')
end

% Reuse figure if it exists, else create new figure
if ~exist('h1_IEEE9BusSystem', 'var') || ...
        ~isgraphics(h1_IEEE9BusSystem, 'figure')
    h1_IEEE9BusSystem = figure('Name', 'IEEE9BusSystem');
end
figure(h1_IEEE9BusSystem)
clf(h1_IEEE9BusSystem)

% Get simulation results
simlog_t = simlog_IEEE9BusSystem.Gen1Bus1_Swing.Rotor_velocity.pu_output.series.time;
simlog_w_Gen1Bus1 = simlog_IEEE9BusSystem.Gen1Bus1_Swing.Rotor_velocity.pu_output.series.values;
simlog_Vt_Gen1Bus1 = simlog_IEEE9BusSystem.Gen1Bus1_Swing.Terminal_voltage.pu_output.series.values;
simlog_w_Gen2Bus2 = simlog_IEEE9BusSystem.Gen2Bus2_PV_1_025_pu_163_MW.Rotor_velocity.pu_output.series.values;
simlog_Vt_Gen2Bus2 = simlog_IEEE9BusSystem.Gen2Bus2_PV_1_025_pu_163_MW.Terminal_voltage.pu_output.series.values;
simlog_w_Gen3Bus3 = simlog_IEEE9BusSystem.Gen3Bus3_PV_1_025_pu_85_MW.Rotor_velocity.pu_output.series.values;
simlog_Vt_Gen3Bus3 = simlog_IEEE9BusSystem.Gen3Bus3_PV_1_025_pu_85_MW.Terminal_voltage.pu_output.series.values;

% Plot results
simlog_handles(1) = subplot(2, 1, 1);
plot(simlog_t, simlog_w_Gen1Bus1, 'LineWidth', 1)
hold on
plot(simlog_t, simlog_w_Gen2Bus2, 'LineWidth', 1)
plot(simlog_t, simlog_w_Gen3Bus3, 'LineWidth', 1)
hold off
grid on
title('Generator rotor speed and terminal voltage')
ylabel('Rotor speed (p.u.)')
legend({'Gen1@Bus1', 'Gen2@Bus2', 'Gen3@Bus3'},'Location','southwest');

simlog_handles(2) = subplot(2, 1, 2);
plot(simlog_t, simlog_Vt_Gen1Bus1, 'LineWidth', 1)
hold on
plot(simlog_t, simlog_Vt_Gen2Bus2, 'LineWidth', 1)
plot(simlog_t, simlog_Vt_Gen3Bus3, 'LineWidth', 1)
hold off
grid on
ylabel('Terminal voltage (p.u.)')
xlabel('Time (s)')
legend({'Gen1@Bus1', 'Gen2@Bus2', 'Gen3@Bus3'},'Location','southwest');
linkaxes(simlog_handles, 'x')

% Remove temporary variables
clear simlog_t simlog_handles
clear simlog_w_Gen1Bus1 simlog_Vt_Gen1Bus1
clear simlog_w_Gen2Bus2 simlog_Vt_Gen2Bus2
clear simlog_w_Gen3Bus3 simlog_Vt_Gen3Bus3
