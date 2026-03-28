clc; clear; close all;
N       = 1e6;                      % Total number of BPSK symbols
EbN0_dB = 0:2:24;                   % Eb/No range in dB
EbN0    = 10.^(EbN0_dB / 10);      % Linear Eb/No

% Pre-allocate BER arrays
BER_ZF   = zeros(1, length(EbN0_dB));
BER_ML   = zeros(1, length(EbN0_dB));
BER_MMSE = zeros(1, length(EbN0_dB));

% BPSK constellation points
bpsk_const = [+1, -1];   % All possible transmitted symbols

bits = randi([0 1], 1, N);
s    = 2*bits - 1;        % Map: 0 -> -1,  1 -> +1

% Pair symbols for 2x2 MIMO (send 2 symbols per time slot)
s1 = s(1:2:end);          % Symbol stream for Tx antenna 1
s2 = s(2:2:end);          % Symbol stream for Tx antenna 2
Np = length(s1);          % Number of time slots

fprintf('Starting simulation with %d symbol pairs...\n', Np);
for idx = 1:length(EbN0_dB)

    noise_var = 1 / (2 * EbN0(idx));   % Noise variance per dimension
    sigma     = sqrt(noise_var);

    % ----- Generate 2x2 Rayleigh fading channel -----
    % h_ij: channel from Tx-j to Rx-i
    h11 = (randn(1,Np) + 1j*randn(1,Np)) / sqrt(2);
    h12 = (randn(1,Np) + 1j*randn(1,Np)) / sqrt(2);
    h21 = (randn(1,Np) + 1j*randn(1,Np)) / sqrt(2);
    h22 = (randn(1,Np) + 1j*randn(1,Np)) / sqrt(2);

    % ----- Received signals at both Rx antennas -----
    % y = H*x + n
    n1 = sigma * (randn(1,Np) + 1j*randn(1,Np));
    n2 = sigma * (randn(1,Np) + 1j*randn(1,Np));

    y1 = h11.*s1 + h12.*s2 + n1;   % Received at Rx antenna 1
    y2 = h21.*s1 + h22.*s2 + n2;   % Received at Rx antenna 2

    % =====================================================
    %  EQUALIZER 1: Zero Forcing (ZF)
    %  W = (H^H * H)^-1 * H^H
    %  Inverts the channel effect, but amplifies noise
    % =====================================================
    s1_zf = zeros(1, Np);
    s2_zf = zeros(1, Np);

    for k = 1:Np
        % Build channel matrix H for this time slot
        H = [h11(k), h12(k);
             h21(k), h22(k)];

        % ZF weight matrix
        W = (H' * H) \ H';         % Equivalent to pinv(H)

        % Received vector
        y = [y1(k); y2(k)];

        % Equalized output
        x_hat = W * y;

        % Hard decision
        s1_zf(k) = sign(real(x_hat(1)));
        s2_zf(k) = sign(real(x_hat(2)));
    end

    errors_zf  = sum(s1_zf ~= s1) + sum(s2_zf ~= s2);
    BER_ZF(idx) = errors_zf / (2 * Np);

    % =====================================================
    %  EQUALIZER 2: Maximum Likelihood (ML)
    %  J = |y - H*x_hat|^2  minimized over all x combos
    %  Exhaustive search over BPSK constellation pairs
    % =====================================================
    % All 4 possible transmitted symbol pairs for BPSK
    candidates = [+1 +1;
                  +1 -1;
                  -1 +1;
                  -1 -1];

    s1_ml = zeros(1, Np);
    s2_ml = zeros(1, Np);

    for k = 1:Np
        H = [h11(k), h12(k);
             h21(k), h22(k)];
        y = [y1(k); y2(k)];

        min_dist = Inf;
        best     = [1, 1];

        for c = 1:4
            x_c  = candidates(c, :).';          % Candidate symbol pair
            diff = y - H * x_c;
            dist = real(diff' * diff);           % Euclidean distance squared

            if dist < min_dist
                min_dist = dist;
                best     = candidates(c, :);
            end
        end

        s1_ml(k) = best(1);
        s2_ml(k) = best(2);
    end

    errors_ml  = sum(s1_ml ~= s1) + sum(s2_ml ~= s2);
    BER_ML(idx) = errors_ml / (2 * Np);

    % =====================================================
    %  EQUALIZER 3: Minimum Mean Square Error (MMSE)
    %  W = (H^H * H + N0*I)^-1 * H^H
    %  Balances noise amplification and ISI suppression
    % =====================================================
    N0 = noise_var * 2;   % Total noise power (real + imag)

    s1_mmse = zeros(1, Np);
    s2_mmse = zeros(1, Np);

    for k = 1:Np
        H = [h11(k), h12(k);
             h21(k), h22(k)];

        % MMSE weight matrix
        W = (H' * H + N0 * eye(2)) \ H';

        y = [y1(k); y2(k)];

        x_hat = W * y;

        s1_mmse(k) = sign(real(x_hat(1)));
        s2_mmse(k) = sign(real(x_hat(2)));
    end

    errors_mmse  = sum(s1_mmse ~= s1) + sum(s2_mmse ~= s2);
    BER_MMSE(idx) = errors_mmse / (2 * Np);

    fprintf('Eb/No = %2d dB | ZF: %.5f | ML: %.5f | MMSE: %.5f\n', ...
        EbN0_dB(idx), BER_ZF(idx), BER_ML(idx), BER_MMSE(idx));
end

% 1x1 SISO BPSK Rayleigh (no diversity, for reference)
BER_theory_1x1 = 0.5 * (1 - sqrt(EbN0 ./ (EbN0 + 1)));

% MRC 1Tx 2Rx theoretical (2nd order diversity, full power per branch)
EbN0_mrc = 2 * EbN0;
BER_theory_mrc = (0.5*(1-sqrt(EbN0_mrc./(EbN0_mrc+1)))).^2 .* ...
                  (1 + 2*sqrt(EbN0_mrc./(EbN0_mrc+1)));
figure('Name','ZF Equalizer','NumberTitle','off',...
       'Color','w','Position',[50 400 700 500]);

semilogy(EbN0_dB, BER_theory_1x1, 'b-s',  'LineWidth',1.5,'MarkerSize',7); hold on;
semilogy(EbN0_dB, BER_theory_mrc, 'k-v',  'LineWidth',1.5,'MarkerSize',7);
semilogy(EbN0_dB, BER_ZF,         'r-o',  'LineWidth',1.5,'MarkerSize',7);

grid on; grid minor;
xlabel('Average Eb/No (dB)', 'FontSize',12);
ylabel('Bit Error Rate',      'FontSize',12);
title('2×2 MIMO — Zero Forcing (ZF) Equalizer','FontSize',13);
legend('theory (nTx=1, nRx=1)', ...
       'theory (nTx=1, nRx=2, MRC)', ...
       'sim   (nTx=2, nRx=2, ZF)', ...
       'Location','southwest','FontSize',10);
axis([0 24 1e-5 1]);
figure('Name','ML Equalizer','NumberTitle','off',...
       'Color','w','Position',[150 250 700 500]);

semilogy(EbN0_dB, BER_theory_1x1, 'b-s',  'LineWidth',1.5,'MarkerSize',7); hold on;
semilogy(EbN0_dB, BER_theory_mrc, 'k-v',  'LineWidth',1.5,'MarkerSize',7);
semilogy(EbN0_dB, BER_ML,         'g-d',  'LineWidth',1.5,'MarkerSize',7);

grid on; grid minor;
xlabel('Average Eb/No (dB)', 'FontSize',12);
ylabel('Bit Error Rate',      'FontSize',12);
title('2×2 MIMO — Maximum Likelihood (ML) Equalizer','FontSize',13);
legend('theory (nTx=1, nRx=1)', ...
       'theory (nTx=1, nRx=2, MRC)', ...
       'sim   (nTx=2, nRx=2, ML)', ...
       'Location','southwest','FontSize',10);
axis([0 24 1e-5 1]);
figure('Name','MMSE Equalizer','NumberTitle','off',...
       'Color','w','Position',[250 100 700 500]);

semilogy(EbN0_dB, BER_theory_1x1, 'b-s',  'LineWidth',1.5,'MarkerSize',7); hold on;
semilogy(EbN0_dB, BER_theory_mrc, 'k-v',  'LineWidth',1.5,'MarkerSize',7);
semilogy(EbN0_dB, BER_MMSE,       'm-p',  'LineWidth',1.5,'MarkerSize',7);

grid on; grid minor;
xlabel('Average Eb/No (dB)', 'FontSize',12);
ylabel('Bit Error Rate',      'FontSize',12);
title('2×2 MIMO — Minimum Mean Square Error (MMSE) Equalizer','FontSize',13);
legend('theory (nTx=2, nRx=2, ZF)', ...
       'theory (nTx=1, nRx=2, MRC)', ...
       'sim   (nTx=2, nRx=2, MMSE)', ...
       'Location','southwest','FontSize',10);
axis([0 24 1e-5 1]);

figure('Name','All Equalizers Comparison','NumberTitle','off',...
       'Color','w','Position',[350 50 750 550]);

semilogy(EbN0_dB, BER_theory_1x1, 'k--',  'LineWidth',1.5); hold on;
semilogy(EbN0_dB, BER_ZF,         'b-o',  'LineWidth',1.5,'MarkerSize',7);
semilogy(EbN0_dB, BER_ML,         'g-d',  'LineWidth',1.5,'MarkerSize',7);
semilogy(EbN0_dB, BER_MMSE,       'r-s',  'LineWidth',1.5,'MarkerSize',7);

grid on; grid minor;
xlabel('Average Eb/No (dB)', 'FontSize',12);
ylabel('Bit Error Rate',      'FontSize',12);
title('2×2 MIMO — ZF vs ML vs MMSE Equalizers','FontSize',13);
legend('theory (nTx=1, nRx=1) — no diversity', ...
       'ZF  (nTx=2, nRx=2)', ...
       'ML  (nTx=2, nRx=2)', ...
       'MMSE(nTx=2, nRx=2)', ...
       'Location','southwest','FontSize',10);
axis([0 24 1e-5 1]);

fprintf('\n============ BER Summary Table ============\n');
fprintf('Eb/No(dB) |   ZF     |   ML     |  MMSE\n');
fprintf('----------|----------|----------|----------\n');
for i = 1:length(EbN0_dB)
    fprintf('  %2d dB   | %.5f  | %.5f  | %.5f\n', ...
        EbN0_dB(i), BER_ZF(i), BER_ML(i), BER_MMSE(i));
end