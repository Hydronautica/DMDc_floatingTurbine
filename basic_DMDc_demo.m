% basic_DMDc_demo.m
% -------------------------------------------------------------
% Corrected Hankel-DMDc (delay embedding d=2) for both state and input
% Uses proper Hankel ordering and direct pseudoinverse (no hard thresholds)
% -------------------------------------------------------------
clear; clc; close all;

%% 1) Generate synthetic data
Tend  = 50;        % total simulation time [s]
dt    = 0.05;      % time step [s]
t     = 0:dt:Tend; % time vector
Ntime = numel(t);

heave =  0.5 * sin(2*pi*0.2*t);            % [m]
pitch =  2.0 * sin(2*pi*0.1*t + pi/6);    % [deg]
surge = -1.0 * cos(2*pi*0.15*t);          % [m]
wave  =  1.2 * sin(2*pi*0.25*t + pi/4);     % [m]

X = [heave; pitch; surge];  % state matrix (3×Ntime)
U = wave;                    % input vector (1×Ntime)

%% 2) Delay embedding dimension
d = 2;                        % number of delays
m = size(X,1);               % state dimension = 3
l = size(U,1);               % input dimension = 1
M = Ntime - d;               % number of Hankel snapshots

%% 3) Build Hankel snapshot matrices
Hx = zeros(m*d, M);  % (3*2)×(Ntime-2)
Hu = zeros(l*d, M);  % (1*2)×(Ntime-2)
Y  = zeros(m,   M);  % 3×(Ntime-2)

for k = 1:M
    % state delays: [ x_k; x_{k+1} ]
    Hx(:,k) = [ X(:,k) ; X(:,k+1) ];
    % input delays: [ u_k; u_{k+1} ]
    Hu(:,k) = [ U(:,k) ; U(:,k+1) ];
    % future state: x_{k+2}
    Y(:,k)  = X(:,k+2);
end

%% 4) Form augmented snapshot and compute pseudoinverse
G = [Hx; Hu];          % (m*d + l*d) × M = 8×M
Gpinv = pinv(G);       % Moore-Penrose pseudoinverse

%% 5) Identify Hankel-DMDc operators
Mbig = Y * Gpinv;      % 3×8 = [ A_d , B_d ]
A_d  = Mbig(:, 1:m*d); % 3×6
B_d  = Mbig(:, m*d+1:end); % 3×2

fprintf('A_d size = %dx%d; B_d size = %dx%d\n', size(A_d), size(B_d));

%% 6) Predict forward using Hankel-DMDc
Xsim = zeros(m, Ntime);
% seed first two true states
Xsim(:,1:d) = X(:,1:d);

for k = (d+1):Ntime
    % build Hankel vector [x_{k-2}; x_{k-1}]
    x_hank = [ Xsim(:,k-2) ; Xsim(:,k-1) ];
    u_hank = [ U(:,   k-2) ; U(:,   k-1) ];
    % predict x_k
    Xsim(:,k) = A_d * x_hank + B_d * u_hank;
end

%% 7) Plot results
figure('Name','Hankel-DMDc Correct (d=2)');
subplot(4,1,1);
plot(t, heave, 'b-', t, Xsim(1,:), 'r--','LineWidth',1.5);
ylabel('Heave [m]'); legend('True','Hankel-DMDc'); grid on;

subplot(4,1,2);
plot(t, pitch, 'b-', t, Xsim(2,:), 'r--','LineWidth',1.5);
ylabel('Pitch [deg]'); grid on;

subplot(4,1,3);
plot(t, surge, 'b-', t, Xsim(3,:), 'r--','LineWidth',1.5);
ylabel('Surge [m]'); grid on;

subplot(4,1,4);
plot(t, wave, 'k-','LineWidth',1.5);
xlabel('Time [s]'); ylabel('Wave [m]'); grid on;
