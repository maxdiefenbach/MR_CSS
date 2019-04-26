close all; clear all; clc;
dataset = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_ImDataParams.mat'

% dataset = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR103_TC/20170824_161342_1202_ImDataParams.mat'

I = ImDataParamsBMRR(dataset)
I.ImDataParams
I.plot

I.load_GradDelayParams
I.set_gradDelayField_rad
I.demodulate_gradDelayField

I.plot

% I.load_B0params
% I.set_magnetInhom_T
% I.set_shimfield_T
% I.plot_B0params

% I.load_ConcomParams
% I.set_concomField_rad
% I.plot_ConcomParams

% I.load_GradDelayParams
% % I.GradDelayParams
% I.set_gradDelayField_rad
% I.demodulate_gradDelayField
% I.save_ImDataParams(strrep(dataset, 'ImDataParams.mat', 'ImDataParams_corr.mat'))
% % I.plot_gradDelayField
% % I.plot_eddyPhaseProfile
% % I.plot_kSpaceShifts

% I.set_ResidualPhaseParams(struct('corrDim', 2))

% TE_s = I.ImDataParams.TE_s
% offset = I.ResidualPhaseParams.kSpaceMaximumOffset_pix


% close all;
% figure
% plot(TE_s(:, 1:2:end), offset(2, 1:2:end))
% hold on
% plot(TE_s(:, 2:2:end), offset(2, 2:2:end))
% legend

% figure
% plot(TE_s(:, 1:2:end), offset(1, 1:2:end))
% hold on
% plot(TE_s(:, 2:2:end), offset(1, 2:2:end))
% legend

% I.set_OBFFMEparams
% I.plot_OBFFMEparams