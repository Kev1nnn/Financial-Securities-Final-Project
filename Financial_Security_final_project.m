%variable 100*\tao (parameter in the Nelson Siegel method)
tao = [100:1000];

len_tao = length(tao);
%zero coupon bond interest rate
M1 = [1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30]; %Maturity for zero-coupon interest rate
s1 = [0.02 0.01 0.02 0.04 0.05 0.16 0.32 0.80 1.27 1.63 2.20 2.32];%interest rate corresponding to each maturity

%obtained from https://www.global-rates.com/en/interest-rates/libor/libor.aspx
M = [1/50,1/12,2/12,3/12,6/12,1]; %Maturity for Libor interest
s = [0.0007150,0.0009813,0.0013250,0.0016750,0.0019250,0.0026700];%Libor rate corresponding to each maturity

% find beta for f by using grid search on tao.
for i = 1:1:len_tao
    tao1 = tao(i)/1000; %calculate from the tao list
    param_T1 = tao1./M.*(1-exp(-M./tao1)); %calculate coefficient for beta_1 with fixed tao
    param_T2 = param_T1-exp(-M./tao1); %calculate coefficient for beta_2 with fixed tao
    X = [param_T1',param_T2']; %set up the domain for linear regresion model
    mdl = fitlm(X,s); %fit the linear model
    
    coef = mdl.Coefficients(:,1:1);%get the estimated coefficients b_0 b_1 b_2 from the model
    stats = anova(mdl,'summary'); %generate the statistics like R^2 or Mean squared error. 
    %initializing the first time coefficient for iteration
    if (i==1)
        best_param = coef.(1); %the best parameter we have so far
        least_err = stats.(3); %the least error we have so far
        best_tao = tao1; %best tao that generate the best parameter combination
        continue;
    end
    
    tmp_err = stats.(3);
    %if we discover a parameter compbination with lower error,select it as
    %temporary parameter selection
    if (tmp_err(3)<least_err(3))
        %update the best parameter
        least_err = tmp_err; 
        best_param = coef.(1);
        best_tao = tao1;
    end
end
%b0=beta_0, b1=beta_1, b2=beta_2 which are parameters in the Nelson Siegel
%model
b0 = best_param(1);
b1 = best_param(2);
b2 = best_param(3);
%paramters initialization
S0 = 271.27211906501435; %S0 is the May 10th 2021 Nikkei-225 index in USD
rf = -0.00138;%risk-free interest rate 
q = 0.0;%dividend is zero for Nikkei-225
T=3; %Maturity
delta = 0.25; %Delta in the problem
cor_sx = -0.3049678; %correlation between equity and exchange rate
sigma_x = 0.0519824811701589; %volatility for exchange rate USD/JPY
sigma_s = 0.199468950979314; %volatility for equity
sigma_r = 0.08; %volatility for short rate
cor_sr = 0.65; %correlation between equity and short rate
workday = 252; %workday- assumed to be 252
dt = 1/workday; %dt just be 1 day
a= 0.03; %mean-reversion factor in the Hull-White Model
equity = zeros(T*workday+1,1); %array that stores simulated equity value at each time step
r = zeros(T*workday+1,1); %array that stores simulated short rate
equity(1) = S0; %the initial value for equity at first day(May 10th 2021)
r(1) = 0.06338; %the initial short rate estimate (using overnight Libor rate)
ytm1 = 0.32; %Yield to maturity with Maturity=3
ytm2 = interp1(M1,s1,T-delta); %Yield to maturity with Maturyti = T-Delta with interpolation
k = 1; %parameter in the Equity portion of max function 
k1 = 1;%parameter in the Libor portion of max function
Num = 1000; %Number of paths
res=0; %the final result
dr_rec = zeros(T*workday+1,1); %array that stores dr at each time step
f_rec = zeros(T*workday,1); %array that stores f (forward rate) at each time step
%loop that generate number of Num paths
for j=1:Num
    %procedure to generate two correlated standard normal random sequence
    e_equity = normrnd(0,1,T*workday,1);
    e_r_before = normrnd(0,1,T*workday,1);
    %by using cholesky decomposition, we get two correlated standard normal
    %random sequence with correlation rho_sr
    e_r = cor_sr.*e_equity+sqrt(1-cor_sr^2).*e_r_before;
    %simulate instantaneous rate and equity difference for each day
    for i= 1:workday*T
        %calculate ds by using Lognormal Equity 
        ds = (rf-q-cor_sx*sigma_s*sigma_x)*dt*equity(i)+sigma_s*sqrt(dt)*e_equity(i)*equity(i);
        equity(i+1) = equity(i)+ds; %record the simulated equity value for tomorrow
        t = i*dt; %calculate the number of days
        %calculate f and df by Nelson and Siegel parametric model
        f = b0+b1*exp(-t/best_tao)+b2*t/best_tao*exp(-t/best_tao);
        f_rec(i) = f;%record the forward rate
        df = 1/best_tao*exp(-t/best_tao)*(-b1+b2*(1-t/best_tao));
        theta = sigma_r^2/(2*a)*(1-exp(-2*a*t))+df+a*f;%calculate theta(t) in Hull-white model
        dr = (theta-a*r(i))*dt+sigma_r*sqrt(dt)*e_r(i); %calculate dr =(\theta(t)-ar)dt+sigma_r\sqrt(dt)e_r)
        dr_rec(i) = dr; %record dr
        r(i+1) = r(i)+dr; %simulated short rate for tomorrow
        %record the value for f(0,T-Delta) and r(T-Delta) for calculation
        %in Libor part
        if (t == T-delta)
            f_Td = f;
            r_Td = r(i);
        end
    end
    p_0T = 1/(1+ytm1/100)^T; %bond price p(0,T) = 1/(1+interest rate)^T
    p_0Td =  1/(1+ytm2/100)^(T-delta); %bond price p(0,T-Delta) = 1/(1+interest rate)^(T-Delta)
    B_Td = 1/a*(1-exp(-a*delta)); %B(T-Delta, T) = 1/a (1-exp(-a*Delta) in Hull-White Model
    %p(T-Delta,T) calculated by using term structure described in the Bjork book
    p_TdT = p_0T/p_0Td*exp(B_Td*f_Td-sigma_r^2/(4*a)*B_Td^2*(1-exp(-2*a*(T-delta)))-B_Td*r_Td); 
    L = 1-p_TdT/(delta*p_TdT); %Libor rate calculation
    payoff = max(0,(equity(T*workday+1)/S0-k)*(k1-L)); %Payoff
    discount = exp(-(sum(r)-r(1))*dt); %discount factor
    res = res+payoff*discount; %sum up all the payoff*discount factor for expectation
end
disp(res/Num); %res/Num is the final expectation that is the answer to the question.