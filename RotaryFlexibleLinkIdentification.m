
% Copyright 2020, All Rights Reserved
% Code by Ghazaale Leylaz
% For Paper, "An Optimal Model Identification Algorithm of 
% Nonlinear Dynamical Systems with the Algebraic Method"
% by Ghazaale Leylaz, Shangjie (Frank) Ma, and Jian-Qiao Sun

clear all
close all

%% Import functions and data folders

addpath('Functions')
addpath('Data')

swEPSfigure
swFigSize

%% Data acquisition:

load('data_alpha');%Alpha
alpha_degree=data_alpha(:,2);
alpha=alpha_degree*pi/180;

load('data_theta');% Theta
theta_degree=data_theta(:,3);
theta=theta_degree*pi/180;

load('data_vm'); %Vm
v=data_vm(:,2);

t=data_vm(:,1);

dt=t(1);
nt=length(t);


%% The system parameters data by Quanser Manual

m_l=0.065;%kg
L_l=0.419;%m
J_l=(m_l*(L_l^2))/3;

K_s=1.3; % gained from natural frequency of the system (experimental value)
B_l=0; % viscous damping of the link

% Servo constants (from its manual)
%B_eq=0.015;%high-gear equivalent viscous damping coefficient
B_eq=0.025;

%J_eq=0.0021;%with load
J_eq=0.002084179192918;% precise value from the devise itself
J_s=K_s*(J_l+J_eq)/J_l;

n_g=0.9; % Gear Efficiency

k_g=70; % High-gear total gear ratio
n_m=0.69; % Motor Efficiency

k_t=7.68e-3;
k_m=7.68e-3;

R_m=2.6; %ohm

a=n_g*k_g*n_m*k_t/R_m;
bb=-n_g*k_g*n_m*k_t*k_g*k_m/R_m;
f1=0; % The acceleration caused by Coulomb friction

% x=[initial theta; initial alpha; initial d_theta; initial d_alpha];
x(:,1)=[0;0;0;0];
dx(:,1)=[0;0;0;0];

% Predicted dynamic response from Quanser's equations and parameters
for n=2:nt
    dx(:,n)=[x(3,n-1); x(4,n-1);((bb+B_l-B_eq)/J_eq)*x(3,n-1)+(K_s*x(2,n-1)-f1)/J_eq+(a/J_eq)*v(n-1);-(((B_l+bb-B_eq)/J_eq)+(B_l/J_l))*x(3,n-1)-J_s*x(2,n-1)/J_eq-(a/J_eq)*v(n-1)];
    x(:,n)=x(:,n-1)+dx(:,n)*dt;
end

theta_theory=x(1,:);
alpha_theory=x(2,:);

%% First Derivative estimation

% compute finite difference derivative as a help to design filter
dtheta(1)=0;
dalpha(1)=0;

ddtheta(1)=0;
ddalpha(1)=0;

for i=2:nt
dtheta(i)=(theta(i)-theta(i-1))/dt;
ddtheta(i)=(dtheta(i)-dtheta(i-1))/dt;

dalpha(i)=(alpha(i)-alpha(i-1))/dt;
ddalpha(i)=(dalpha(i)-dalpha(i-1))/dt;
end

%%% filtering approach

sigma_theta=0.05; % gain for theta signal
sigma_alpha=0.05; % gain for alpha signal

est_theta = nleso(theta',dt,sigma_theta);
thetaf = est_theta(1,:)';
dthetaf= est_theta(2,:)';


est_alpha = nleso(alpha',dt,sigma_alpha);
alphaf = est_alpha(1,:)';
dalphaf= est_alpha(2,:)';


%%% plot
figure()
subplot(2,2,1)
plot(t,theta,t,thetaf)
ylabel('theta')
subplot(2,2,3)
plot(t,dtheta,t,dthetaf)
ylabel('dtheta')


subplot(2,2,2)
plot(t,alpha,t,alphaf)
ylabel('alpha')
subplot(2,2,4)
plot(t,dalpha,t,dalphaf)
ylabel('dalpha')


%% Library generation, split data and model searching

% split data
ratio=0.5; % 50 % for training, 50 % for cross validation (AIC calculation)
ntr=round(ratio*nt);
nts=nt-ntr;

vAl=Al(2,2,v,t);
vAl_tr=vAl(1:ntr,1);
vAl_ts=vAl(ntr+1:end,1);


% K-fold Lasso Crossvalidation
kfold=10;

% generate Lambda
numlambda = 50;
lambdastart = -4;
lambdaend = -1;
Lambda = logspace(lambdastart,lambdaend, numlambda);

% setting for generate data library for each model order 
Algebraic=1; 
CrossedProducts=0;  

CoulombFrictionTheta=1;
CoulombFrictionAlpha=0;

%
MaxPolTheta=4;
MaxPolAlpha=4;

parameters=cell(MaxPolTheta,MaxPolAlpha);

 for polyorderTheta=1:MaxPolTheta
     for polyorderAlpha=1:MaxPolAlpha
         
     clear pTheta pAlpha p1 p2 p1_tr p2_tr p1_ts p2_ts prms1 prms2 ...
         Eq1prms Eq2prms MSEcvLambda1 MSEcvLambda2 MSEcv_min1 MSEcv_min2 
     
     pTheta=Lib(theta,dthetaf,ddtheta,t,polyorderTheta,CrossedProducts,Algebraic,CoulombFrictionTheta);
     pAlpha=Lib(alpha,dalphaf,ddalpha,t,polyorderAlpha,CrossedProducts,Algebraic,CoulombFrictionAlpha);
     
     % organize generated library to a neety shape 
     p1=[pTheta(:,1:end-1),pAlpha(:,2:end),pTheta(:,end)];
     p2=[pAlpha(:,1),pTheta(:,2:end-1),pAlpha(:,2:end),pTheta(:,end)];
     
     % split data for traing and cross validation for model selection AIC
     p1_tr=p1(1:ntr,:);
     p2_tr=p2(1:ntr,:);
     
     p1_ts=p1(ntr+1:end,:);
     p2_ts=p2(ntr+1:end,:);

     
     % Sparse regression with K-fold cross validation
     for i=1:length(Lambda)
         [MSEcvLambda1(i),~]=lassoKfold(p1_tr,vAl_tr,Lambda(i),kfold,0);
         [MSEcvLambda2(i),~]=lassoKfold(p2_tr,vAl_tr,Lambda(i),kfold,0);
     end
     
     MSEcvTheta{polyorderTheta,polyorderAlpha}=MSEcvLambda1;
     MSEcvAlpha{polyorderTheta,polyorderAlpha}=MSEcvLambda2;

     % find regulator (tuning) parameter lambda
     % Theta equation
     [MSEcv_min1,IndexMSEcvmin1]=min(MSEcvLambda1);
     lambdaMinMSEcv1{polyorderTheta,polyorderAlpha} = Lambda(IndexMSEcvmin1);
     Eq1prms=sparsifyDynamics(p1_tr,vAl_tr,lambdaMinMSEcv1{polyorderTheta,polyorderAlpha},1);
     K1=nnz(Eq1prms);
     %%% CV1(polyorderTheta,polyorderAlpha)=MSEcv_min1;
     RSS1(polyorderTheta,polyorderAlpha)=(p1_ts*Eq1prms-vAl_ts)'*(p1_ts*Eq1prms-vAl_ts);
     AIC1(polyorderTheta,polyorderAlpha)=(nts*log(RSS1(polyorderTheta,polyorderAlpha)/nts))+2*K1;
     AICc1(polyorderTheta,polyorderAlpha)=AIC1(polyorderTheta,polyorderAlpha)+((2*(K1+1)*(K1+2))/(nts-K1-2));
     BIC1(polyorderTheta,polyorderAlpha)=(nts*log(RSS1(polyorderTheta,polyorderAlpha)/nts))+2*K1*log(nts);

     % Alpha equation
     [MSEcv_min2,IndexMSEcvmin2]=min(MSEcvLambda2);
     lambdaMinMSEcv2{polyorderTheta,polyorderAlpha} = Lambda(IndexMSEcvmin2);
     Eq2prms=sparsifyDynamics(p2_tr,vAl_tr,lambdaMinMSEcv2{polyorderTheta,polyorderAlpha},1);
     K2=nnz(Eq2prms);
     %%% CV2(polyorderTheta,polyorderAlpha)=MSEcv_min2;
     RSS2(polyorderTheta,polyorderAlpha)=(p2_ts*Eq2prms-vAl_ts)'*(p2_ts*Eq2prms-vAl_ts);
     AIC2(polyorderTheta,polyorderAlpha)=(nts*log(RSS2(polyorderTheta,polyorderAlpha)/nts))+2*K2;
     AICc2(polyorderTheta,polyorderAlpha)=AIC2(polyorderTheta,polyorderAlpha)+((2*(K2+1)*(K2+2))/(nts-K2-2));
     BIC2(polyorderTheta,polyorderAlpha)=(nts*log(RSS2(polyorderTheta,polyorderAlpha)/nts))+2*K2*log(nts);
     
     parameters{polyorderTheta,polyorderAlpha}=[Eq1prms,Eq2prms]; % save paramters for each model
     
     end
     
 end
 
%% Saving Data
save('RotaryFlexibleBeam.mat');

%% 2D plotting

figure
[AIC1_min_2d,AIC1_min_Index_2d]=min(diag(AIC1,0));
[AIC2_min_2d,AIC2_min_Index_2d]=min(diag(AIC2,0));

subplot(2,1,1)
plot(1:polyorderTheta,diag(AIC1,0),'-O','LineWidth',1.5)
xlabel('Polynomial Order')
ylabel('$AIC_{\theta}$')
hold on
plot(AIC1_min_Index_2d,AIC1_min_2d,'or','LineWidth',3)
grid on
% set(gca, 'YScale', 'log')

subplot(2,1,2)
plot(1:polyorderAlpha,diag(AIC2,0),'-O','LineWidth',1.5)
xlabel('Polynomial Order')
ylabel('$AIC_{\alpha}$')
hold on
plot(AIC2_min_Index_2d,AIC2_min_2d,'or','LineWidth',3)
grid on
% set(gca, 'YScale', 'log')

print -depsc AICRotaryFlexibleLink.eps

%% Predict response
%plot response for the order 3,2
prms=parameters{3,2}
prms1=prms(:,1);
prms2=prms(:,2);

%% Regularization (tuning) parameter lambda plot for the true model
figure
subplot(2,1,1)
plot(Lambda,MSEcvTheta{3,2},'LineWidth',1.5)
xline(lambdaMinMSEcv1{3,2},'-.b','\lambda_{\theta,min}','LineWidth',1.5,'FontSize', 16);
ylabel('$MSEcv-{\theta}$')
grid on
subplot(2,1,2)
plot(Lambda,MSEcvAlpha{3,2},'LineWidth',1.5)
xline(lambdaMinMSEcv2{3,2},'-.b','\lambda_{\alpha,min}','LineWidth',1.5,'FontSize', 16);
ylabel('$MSEcv-{\alpha}$')
xlabel('$\lambda$')
grid on
print -depsc LambdaSelectionRotaryFlexibleLink.eps

%% Plot the predicted response of the identified model in comparisoon with
% Quanser Manual
%x=[initial theta; initial alpha; initial d_theta; initial d_alpha];
x_pr(:,1)=[0;0;0;0];
dx_pr(:,1)=[0;0;0;0];

for n=2:nt
    dx_pr(:,n)=[x_pr(3,n-1);
        
        x_pr(4,n-1);
        
        (v(n-1)-prms1(2)*x_pr(1,n-1)-prms1(3)*(x_pr(1,n-1)^2)-prms1(4)*(x_pr(1,n-1)^3)...
        -prms1(5)*x_pr(3,n-1)-prms1(6)*(x_pr(3,n-1)^2)-prms1(7)*(x_pr(3,n-1)^3)...
        -prms1(8)*x_pr(2,n-1)-prms1(9)*(x_pr(2,n-1)^2)...
        -prms1(10)*x_pr(4,n-1)-prms1(11)*(x_pr(4,n-1)^2)...
        -prms1(end)*sign(x_pr(3,n-1)))/prms1(1);
        
        (v(n-1)-prms2(2)*x_pr(1,n-1)-prms2(3)*(x_pr(1,n-1)^2)-prms2(4)*(x_pr(1,n-1)^3)...
        -prms2(5)*x_pr(3,n-1)-prms2(6)*(x_pr(3,n-1)^2)-prms2(7)*(x_pr(3,n-1)^3)...
        -prms2(8)*x_pr(2,n-1)-prms2(9)*(x_pr(2,n-1)^2)...
        -prms2(10)*x_pr(4,n-1)-prms2(11)*(x_pr(4,n-1)^2)...
        -prms2(end)*sign(x_pr(3,n-1)))/prms2(1)];
    
    x_pr(:,n)=x_pr(:,n-1)+dx_pr(:,n)*dt;
end

theta_pr=x_pr(1,:);
alpha_pr=x_pr(2,:);

%%% plotting
figure()

subplot(311)
plot(t,v,'LineWidth',1.5)
ylabel('Voltage (v)')
grid on

subplot(312)
plot(t,theta_theory,t,theta,'-o','MarkerIndices',1:50:length(t),'LineWidth',1.5)
hold on
plot(t,theta_pr,'-x','MarkerIndices',1:50:length(t),'LineWidth',1.5)

ylabel('$\theta$ (rad)')
legend('Theo.','Exp.','Alg. ID');
grid on

subplot(313)
plot(t,alpha_theory,t,alpha,'-o','MarkerIndices',1:50:length(t),'LineWidth',1.5)

hold on
plot(t,alpha_pr,'-x','MarkerIndices',1:50:length(t),'LineWidth',1.5)

xlabel('Time (s)')
ylabel('$\alpha$ (rad)')

grid on

print -depsc SystemResponseComparison.eps


