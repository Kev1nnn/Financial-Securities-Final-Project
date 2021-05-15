
tao = [1:1000];

len_tao = length(tao);

M1 = [1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30];
s1 = [0.02 0.01 0.02 0.04 0.05 0.16 0.32 0.80 1.27 1.63 2.20 2.32];
%obtained from https://www.global-rates.com/en/interest-rates/libor/libor.aspx
M = [1/50,1/12,2/12,3/12,6/12,1];
s = [0.0007150,0.0009813,0.0013250,0.0016750,0.0019250,0.0026700];
% find beta for f by using grid search on tao.
for i = 1:1:len_tao
    tao1 = tao(i)/100;
    param_T1 = tao1./M.*(1-exp(-M./tao1));
    param_T2 = param_T1-exp(-M./tao1);
    X = [param_T1',param_T2'];
    mdl = fitlm(X,s);
    
    coef = mdl.Coefficients(:,1:1);
    stats = anova(mdl,'summary');
    %initializing the first time coefficient for iteration
    
    if (i==1)
        best_param = coef.(1);
        least_err = stats.(3);
        best_tao = tao1;
        continue;
    end
    %if we find a combination of paramters that has least error, choose it.
    tmp_err = stats.(3);
    if (tmp_err(3)<least_err(3))
        least_err = tmp_err;
        best_param = coef.(1);
        best_tao = tao1;
    end
end

b0 = best_param(1);
b1 = best_param(2);
b2 = best_param(3);

%paramters initialization
S0 = 271.27211906501435;
rf = -0.00138;
q = 0.0;
T=3;
delta = 0.25;
cor_sx = -0.3049678;
sigma_x = 0.0519824811701589;
sigma_s = 0.199468950979314;
cor_sr = 0.65;
workday = 252;
dt = 1/workday;
a= 0.03;
sigma_r = 0.08;
equity = zeros(T*workday+1,1);
r = zeros(T*workday+1,1);
equity(1) = S0;
r(1) = 0.06338; %whether use the value with percentage or not?
ytm1 = 0.05;
ytm2 = interp1(M1,s1,T-delta);
k = 0.2; %two parameter in the question
k1 = 0.5;
Num = 1000; %Number of paths
res=0;
for j=1:Num
    e_equity = normrnd(0,1,T*workday,1);
    e_r_before = normrnd(0,1,T*workday,1);
    e_r = cor_sr.*e_equity+sqrt(1-cor_sr^2).*e_r_before;
    for i= 1:workday*T
        %calculate ds by using Lognormal Equity 
        ds = (rf-q-cor_sx*sigma_s*sigma_x)*dt*equity(i)+sigma_s*sqrt(dt)*e_equity(i)*equity(i);
        equity(i+1) = equity(i)+ds;
        t = i*dt;
        %calculate f and df by Nelson and Siegel parametric model
        f = b0+b1*exp(-t/best_tao)+b2*t/best_tao*exp(-t/best_tao);
        df = 1/best_tao*exp(-t/best_tao)*(-b1+b2*(1-t/best_tao));
        theta = sigma_r^2/(2*a)*(1-exp(-2*a*t))+df+a*f;
        dr = (theta-a*r(i))*dt+sigma_r*sqrt(dt)*e_r(i);
        r(i+1) = r(i)+dr;
        if (t == T-delta)
            f_Td = f;
            r_Td = r(i);
        end
    end

    p_0T = 1/(1+ytm1/100);
    p_0Td =  1/(1+ytm2/100)^0.75;
    B_Td = 1/a*(1-exp(-a*delta));
    p_TdT = p_0T/p_0Td*exp(B_Td*f_Td-sigma_r^2/(4*a)*B_Td^2*(1-exp(-2*a*(T-delta)))-B_Td*r_Td);
    L = 1-p_TdT/(delta*p_TdT);
    payoff = max(0,(equity(T*workday+1)/S0-k)*(k1-L));
    discount = exp(-(sum(r)-r(1))*dt);
    res = res+payoff*discount;
end
disp(res/Num);