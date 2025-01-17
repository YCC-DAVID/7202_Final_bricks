clear;clc;clf
returns_cartpole_pg_r = readtable('returns_cartpole_pg_r_revise.csv');
returns_cartpole_pg_r = table2array(returns_cartpole_pg_r);
returns_cartpole_pg_r = returns_cartpole_pg_r(1:100, :);

returns_cartpole_dqn = readtable('returns_cartpole_dqn.csv');
returns_cartpole_dqn = table2array(returns_cartpole_dqn);
returns_cartpole_dqn = returns_cartpole_dqn(1:100, :);

returns_cartpole_pg = readtable('returns_cartpole_pg.csv');
returns_cartpole_pg = table2array(returns_cartpole_pg);
returns_cartpole_pg = returns_cartpole_pg(1:100, :);

returns_cartpole_pg = movmean(returns_cartpole_pg, 5);
returns_cartpole_pg_r = movmean(returns_cartpole_pg_r, 5);
returns_cartpole_dqn = movmean(returns_cartpole_dqn, 5);

figure('Position', [100, 100, 800, 600]);

hold on;
plot(returns_cartpole_pg_r);
title('Returen in DQN and PG (Cartpole)');

plot(returns_cartpole_dqn);
legend('PG Rewards', 'DQN Rewards');

ylim([0, 400]);
xlabel Episode
ylabel Return

figure('Position', [100, 100, 800, 600]);
subplot(1,2,1);
plot(returns_cartpole_pg);
title('Returen PG (Cartpole)');
ylim([0, 200]);
xlabel Episode
ylabel Return

subplot(1,2,2);
plot(returns_cartpole_pg_r);
title('Returen Re\_PG (Cartpole)');
ylim([0, 200]);
xlabel Episode
ylabel Return
