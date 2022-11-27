%% This function draw figures
clear all; close all
%{
A = dir(fullfile('./Figure/*'));
if ~isempty(A)
    for k = 1:length(A)
        delete(strcat('./Figure/', A(k).name))
    end
    fprintf('remove result file. Done!\n')
else
    fprintf('No result file.\n')
end
%}
load('./Solution/AdapSQP.mat')
load('./Solution/AdapL1SQP.mat')
load('./Solution/BerahasSQP.mat')
load('./Solution/NonAdapSQP.mat')

%% Plot
% global color
col = horzcat(hsv(4),ones(4,1)*0.5)';
col2 = col(:,3:4); 

%% Plot KKT residual with constant 1
for step = 1:6 
    for sigma = 1:5
        AR{sigma} = Res{sigma,1}.KKT;
        BR{sigma} = ResL1{sigma,1}.KKT;
        CR{sigma} = ResB{step,sigma}.KKT;
        if length(ResN{step,sigma}.KKT)>0
            DR{sigma} = ResN{step,sigma}.KKT;
        else
            DR{sigma} = [NaN];
        end 
    end
    data=vertcat(AR,BR,CR,DR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapSQP','$\ell_1$ AdapSQP','$\ell_1$ SQP', 'NonAdapSQP'};
    figure(step)
    multiple_boxplot(data,xlab,Mlab,col,1,0)
%    filename = ['./Figure/KKTStep' num2str(step) '.png'];
%    print('-dpng', filename)
end

%% Plot KKT residual with varying constant
for cons = 1:4 
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.KKT;
        BR{sigma} = ResL1{sigma,cons}.KKT;
    end
    data=vertcat(AR,BR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapSQP','$\ell_1$ AdapSQP'};
    figure(10+cons)
    multiple_boxplot(data,xlab,Mlab,col2,1.5,0)
%    filename = ['./Figure/KKTCons' num2str(cons) '.png'];
%    print('-dpng', filename)
end

%% Plot Sample size
for cons = 1:4
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.CountG;
        BR{sigma} = ResL1{sigma,cons}.CountG;
        CR{sigma} = ResB{1,sigma}.Count;
        if length(ResN{1,sigma}.Count)>0
           DR{sigma} = 2*ResN{1,sigma}.Count;
        else
           DR{sigma} = [NaN];
        end
    end
    data=vertcat(AR,BR,CR,DR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapSQP','$\ell_1$ AdapSQP','$\ell_1$ SQP', 'NonAdapSQP'};
    figure(100+cons)
    multiple_boxplot(data,xlab,Mlab,col,2,0)    
%    filename = ['./Figure/GSampleCons' num2str(cons) '.png'];
%    print('-dpng', filename)
end


for cons = 1:4
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.CountF;
        BR{sigma} = ResL1{sigma,cons}.CountF;
    end
    data=vertcat(AR,BR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapSQP','$\ell_1$ AdapSQP'};
    figure(200+cons)
    multiple_boxplot(data,xlab,Mlab,col2,3,0)    
%    filename = ['./Figure/FSampleCons' num2str(cons) '.png'];
%   print('-dpng', filename)
end

%% Plot Stepsize
cmap = jet(10);
for sigma = 1:5
    figure(1000)
    subplot(5,1,sigma)
    ProbId = Res{sigma,1}.ProbId;
    for ii = 1:10
        plot(Res{sigma,1}.Alpha{ProbId(ii)}, 'Color', cmap(ii,:),'LineWidth',1)
        set(gca,'fontsize',14)
        hold on 
    end
    xLimits = get(gca,'XLim');    
    line(xLimits,[1,1],'Color','black','LineStyle','--')
    hold off
%    filename = ['./Figure/StepA'  '.png'];
%    print('-dpng', filename)   
end     
    
for sigma = 1:5
    figure(1001)
    subplot(5,1,sigma)
    ProbId = ResL1{sigma,1}.ProbId;
    for ii = 1:10
        plot(ResL1{sigma,1}.Alpha{ProbId(ii)}, 'Color', cmap(ii,:),'LineWidth',1)
        set(gca,'fontsize',14)
        hold on 
    end
    xLimits = get(gca,'XLim');    
    line(xLimits,[1,1],'Color','black','LineStyle','--')
    hold off
%    filename = ['./Figure/StepAL1' '.png'];
%    print('-dpng', filename)
end     



%% Output convergence status
ProbId = fopen('./Parameter/problems.txt','r');
Prob = textscan(ProbId,'%s','delimiter','\n');

for sigma = 1:5 
    for Idprob = 1:47
        % result of AdapSQP
        a = [];b = [];
        for cons = 1:4
            xindex = find(Res{sigma,cons}.ProbId == Idprob);
            if xindex 
                a = [a; Res{sigma,cons}.KKT(xindex)];
                b = [b; Res{sigma,cons}.Std(xindex)];                
            end
        end
        if ~isempty(a)
            [~, id] = min(a);
            Ra = [1, log(min(a)), log(b(id))];
        else 
            Ra = [0, 100, 0];
        end
        
        % result of AdapSQPL1
        a = [];b = [];
        for cons = 1:4
            xindex = find(ResL1{sigma,cons}.ProbId == Idprob);
            if xindex 
                a = [a; ResL1{sigma,cons}.KKT(xindex)];
                b = [b; ResL1{sigma,cons}.Std(xindex)];                
            end
        end
        if ~isempty(a) && min(a)<10^5
            [~, id] = min(a);
            Rb = [1, log(min(a)), log(b(id))];
        else 
            Rb = [0, 100, 0];
        end
        
        % result of BerahasSQP
        a = [];b = [];
        for step = 1:6
            xindex = find(ResB{step,sigma}.ProbId == Idprob);
            if xindex 
                a = [a; ResB{step,sigma}.KKT(xindex)];
                b = [b; ResB{step,sigma}.Std(xindex)];
            end
        end
        if ~isempty(a) && ~isnan(min(a))
            [~, id] = min(a);
            Rc = [1, log(min(a)), log(b(id))];
        else
           Rc = [0, 100, 0];
        end
        
        % result of NonAdapSQP
        a = [];b = [];
        for step = 1:6
            xindex = find(ResN{step,sigma}.ProbId == Idprob);
            if xindex 
                a = [a; ResN{step,sigma}.KKT(xindex)];
                b = [b; ResN{step,sigma}.Std(xindex)];
            end
        end    
        if ~isempty(a) && ~isnan(min(a))
            [~, id] = min(a);
            Rd = [1, log(min(a)), log(b(id))];
        else
           Rd = [0, 100, 0];
        end
        
%{ 
        filename = ['./Figure/sigma' num2str(sigma) '.txt'];
        if exist(filename, 'file') == 0
            f = fopen(filename, 'w+');
            fprintf(f, '%s', Prob{1}{Idprob});
            fprintf(f, ' [%3d, %5.2f, %7.2f]', Ra);
            fprintf(f, '[%8d, %10.2f, %12.2f]', Rb);
            fprintf(f, '[%13d, %15.2f, %17.2f]', Rc);
            fprintf(f, '[%18d, %20.2f, %22.2f]\n', Rd);
        else
            f = fopen(filename, 'a+');
            fprintf(f, '%s', Prob{1}{Idprob});
            fprintf(f, ' [%3d, %5.2f, %7.2f]', Ra);
            fprintf(f, '[%8d, %10.2f, %12.2f]', Rb);
            fprintf(f, '[%13d, %15.2f, %17.2f]', Rc);
            fprintf(f, '[%18d, %20.2f, %22.2f]\n', Rd);
        end
%}        
    end
end



