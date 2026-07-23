clc;

% Data loading   
path = "/vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp/"

%T = load(path + "" + "BlackLeggedKittiwake.mat");
%[momp_out, momp_loc] = momp_v9(T.BlackLeggedKittiwake, 512, 1)

%T = load(path + "" + "MGHSleepElectromyography.mat");
%[momp_out, momp_loc] = momp_v9(T.MGHSleepElectromyography, 512, 1)

%T = load(path + "swtAttack7" + ".mat");
%[momp_out, momp_loc] = momp_v9(T.swtAttack7, 512, 1)

%T = load(path + "" + "HAR_Ambient_Sensor_Data.mat");
%[momp_out, momp_loc] = momp_v9(T.HAR_Ambient_Sensor_Data, 512, 1)

%T = load(path + "" + "Lab_FD_061014.mat");
%[momp_out, momp_loc] = momp_v9(T.volts, 512, 1)

%T = load(path + "" + "swtAttack38.mat");
%[momp_out, momp_loc] = momp_v9(T.swtAttack38, 512, 1)

%T = load(path + "" + "house.mat");
%[momp_out, momp_loc] = momp_v9(T.house, 512, 1)

%T = load(path + "" + "WindTurbine.mat");
%[momp_out, momp_loc] = momp_v9(T.WindTurbine, 512, 1)

%T = load(path + "" + "water.mat");
%[momp_out, momp_loc] = momp_v9(T.water, 512, 1)

%T = load(path + "" + "Challenge2009Respiration500HZ.mat");
%[momp_out, momp_loc] = momp_v9(T.Challenge2009Respiration500HZ, 512, 1)

%T = load(path + "EOG_one_hour_50_Hz" + ".mat");
%[momp_out, momp_loc] = momp_v9(T.EOG_one_hour_50_Hz, 512, 1)

%T = load(path + "" + "solarwind.mat");
%[momp_out, momp_loc] = momp_v9(T.solarwind, 512, 1)

%T = load(path + "" + "SpainishEnergyDataset5sec.mat");
%[momp_out, momp_loc] = momp_v9(T.SpainishEnergyDataset5sec, 512, 1)

%T = load(path + "" + "Challenge2009TestSetA_101a.mat");
%[momp_out, momp_loc] = momp_v9(T.Challenge2009TestSetA_101a, 512, 1)

%T = load(path + "" + "Lab_K_060314.mat");
%[momp_out, momp_loc] = momp_v9(T.volts, 512, 1)

%T = load(path + "" + "stator_winding.mat");
%[momp_out, momp_loc] = momp_v9(T.stator_winding, 512, 1)

%T = load(path + "EOG_one_hour_400_Hz" + ".mat");
%[momp_out, momp_loc] = momp_v9(T.EOG_one_hour_400_Hz,512, 1)

%T = load(path + "CinC_Challenge" + ".mat");
%[momp_out, momp_loc] = momp_v9(T.CinC_Challenge, 512, 1)

T = load(path + "" + "recorddata.mat");
[momp_out, momp_loc] = momp_v9(T.recorddata(3,1:1:end), 512, 1)

T = load(path + "" + "Bird12-Week3_2018_1_10.mat");
[momp_out, momp_loc] = momp_v9(T.bird1, 512, 1)

T = load(path + "" + "FingerFlexionECoG.mat");
[momp_out, momp_loc] = momp_v9(T.FingerFlexionECoG, 512, 1)

T = load(path + "" + "lorenzAttractorsLONG.mat");
[momp_out, momp_loc] = momp_v9(T.lorenzAttractorsLONG, 512, 1)

T = load(path + "" + "SynchrophasorEventsLarge.mat");
[momp_out, momp_loc] = momp_v9(T.SynchrophasorEventsLarge, 512, 1)

