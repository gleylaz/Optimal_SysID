
% Copyright 2020, All Rights Reserved
% Code by Ghazaale Leylaz
% For Paper, "An Optimal Model Identification Algorithm of 
% Nonlinear Dynamical Systems with the Algebraic Method"
% by Ghazaale Leylaz, Shangjie (Frank) Ma, and Jian-Qiao Sun

%% import functions and data folders

addpath('Functions')
addpath('Data')

% add plot settings
swEPSfigure
swFigSize

%%
% System paratameters for modeling the system:
% mddx=-k1*x-k2*x^2-k3*x^3-c1*x-c2*x^2*dx-c3*dx^3
clear all;
close all;

m=2;

k1=6;
k2=0;
k3=0.1;

c1=1;
c2=0.2;
c3=1;

t_initial=0;
dt=0.001;
t_final=20;
tout=t_initial:dt:t_final;
tout=tout';
nt=length(tout);


for i=1:nt
    Force(i)=10*sin(tout(i))+40*sin(4*tout(i));
end

% [initial position;initial velocity];
x(:,1)=[0;0];
dx(:,1)=[0;0];

% Euler method to solve the motion equation
for n=2:nt
    dx(:,n)=[x(2,n-1);(Force(n-1)-k1*x(1,n-1)^1-k2*(x(1,n-1)^2)-k3*(x(1,n-1)^3)-c1*x(2,n-1)-c2*x(2,n-1)*(x(1,n-1)^2)-c3*(x(2,n-1)^3))/m];
    x(:,n)=x(:,n-1)+dx(:,n)*dt;
end

v=x(2,:)';% speed
a=dx(2,:)';%acceleration
x=x(1,:)';%displacement

epsX=0.01;
epsF=0.01;

xnoisevec = epsX*randn(nt,1);
x=x+xnoisevec;

fnoisevec = epsF*randn(nt,1);
Force=Force'+fnoisevec;

% figure()
% plot(tout,x);
% grid on
% xlabel('time')
% ylabel('Displacement')
% title('Displacement Vs Time') 
% 
% % Derivatives Estimations
% dX(1)=0;
% ddX(1)=0;
% 
% for i=2:nt
% dX(i)=(x(i)-x(i-1))/dt;
% ddX(i)=(dX(i)-dX(i-1))/dt;
% end

%%
Pf=Al(2,2,Force,tout);

% Number of folds
kfold=5;

% make a vector of threshold values
numlambda = 100;
lambdastart = -8;
lambdaend = 0;
Lambda = logspace(lambdastart,lambdaend, numlambda);


% split data
ratio=0.5;
ntr=round(ratio*nt);
nts=nt-ntr;

Pf_tr=Pf(1:ntr,1);
Pf_ts=Pf(ntr+1:end,1);


%%
MaxPol=5;% Maximum polynomial order to study
parameters=cell(1,MaxPol);

for polyorder=1:MaxPol
    clear p_model MSEcvLambda prms optimalKfold lambdaMinMSEcv K MSEcv_min IndexMSEcvmin 
    % Make the data library by the function
    % Lib(x,dx,ddx,t,polyorder,CrossedProducts,Algebraic,CoulombFriction)
    p_model=Lib(x,v,a,tout,polyorder,1,1,0);
    
    % split
    p_tr=p_model(1:ntr,:);
    p_ts=p_model(ntr+1:end,:);
    
    % K-fold cross-validation
    for i=1:length(Lambda)
        [MSEcvLambda(i),~]=lassoKfold(p_tr,Pf_tr,Lambda(i),kfold,0);
    end
    
    MSEcv{polyorder}=MSEcvLambda;
    % Model selection
   [MSEcv_min,IndexMSEcvmin]=min(MSEcvLambda);
   lambdaMinMSEcv = Lambda(IndexMSEcvmin);
   
   optimalKfold=sparsifyDynamics(p_tr,Pf_tr,lambdaMinMSEcv,1);
   parameters{polyorder}=optimalKfold;
   K=nnz(optimalKfold);
   
   RSS(polyorder)=(p_ts*optimalKfold-Pf_ts)'*(p_ts*optimalKfold-Pf_ts);
   AIC(polyorder)=(nts*log(RSS(polyorder)/nts))+2*K;

end

%% Saving Data
save('NonlinearMCK.mat');

%% AIC Plot
figure()
plot(1:polyorder,AIC,'-O','LineWidth',1.5)
xlabel('Polynomial Order')
ylabel('AIC')
[AIC_min,AIC_min_Index]=min(AIC);
hold on
plot(AIC_min_Index,AIC_min,'or','LineWidth',3)
xline(AIC_min_Index,'-.b','LineWidth',1.5,'FontSize', 16);
grid on
print -depsc AICmck.eps

%% Parameters estimations for the selected order

True_order=AIC_min_Index;

Estimated_Parameters=parameters{True_order}

%% Regularization (tuning) parameter lambda plot for the true model

figure()
plot(Lambda,MSEcv{True_order},'LineWidth',1.5)

[MSEcv_min,IndexMSEcvmin]=min(MSEcv{True_order});
lambdaMinMSEcv = Lambda(IndexMSEcvmin);

xlabel('$\lambda$')
xline(lambdaMinMSEcv,'-.b','\lambda_{min}','LineWidth',1.5,'FontSize', 24);
ylabel('$MSEcv$')
grid on
print -depsc LambdaSelectionMCK.eps