%% This fucntion draws plot for simulation
clear all; close all
%d = dir('./Solution/*.mat');
%A = dir(fullfile('./Figure/*'));
%if ~isempty(A)
%    for k = 1:length(A)
%        delete(strcat('./Figure/', A(k).name))
%    end
%    fprintf('remove result file. Done!\n')
%else
%    fprintf('No result file.\n')
%end

%% Extract AdapSQP result
for sigma = 1:5
    KKTVec = [];
    TimeVec = [];
    CountVec = [];
    for Idprob = 1:47
        ll = length(AdapR{Idprob}.KKTStep{sigma});
        if ll > 0
            a = [];
            b = [];
            c = [];
            for lll = 1:ll 
                a = [a; AdapR{Idprob}.KKTStep{sigma}{lll}(end)];
                b = [b; AdapR{Idprob}.CountStep{sigma}{lll}(end)];
                c = [c; AdapR{Idprob}.TimeStep{sigma}{lll}(end)];
            end
            KKTVec = [KKTVec; min(a)];
            CountVec = [CountVec; min(b)];
            TimeVec = [TimeVec; min(c)];
        end 
    end 
    AdapKKT{sigma} = KKTVec;
    AdapCount{sigma} = CountVec;
    AdapTime{sigma} = TimeVec;
end


%% Extract BerahasSQP result 
BerahasR = [BerahasR1;BerahasR2];
for ConStep = 1:4
    for sigma = 1:5 
        KKTVec = [];
        TimeVec = [];
        CountVec = [];
        for Idprob = 1:47
            ll = length(BerahasR{Idprob}.KKTC{ConStep,sigma});
            if ll > 0 
                a = [];
                b = [];
                c = [];
                for lll = 1:ll 
                    a = [a; BerahasR{Idprob}.KKTC{ConStep,sigma}{lll}(end)];
                    b = [b; BerahasR{Idprob}.TimeC{ConStep,sigma}{lll}(end)];
                    c = [c; length(BerahasR{Idprob}.KKTC{ConStep,sigma}{lll})];
                end
                if ~isnan(min(a))
                    KKTVec = [KKTVec; min(a)];
                    TimeVec = [TimeVec; min(b)];
                    CountVec = [CountVec; min(c)];
                end
            end 
        end
        BerhasCKKT{ConStep,sigma} = KKTVec;
        BerhasCTime{ConStep,sigma} = TimeVec;
        BerhasCCount{ConStep,sigma} = CountVec;
    end 
end

for DecayStep = 1:2
    for sigma = 1:5 
        KKTVec = [];
        TimeVec = [];
        CountVec = [];
        for Idprob = 1:47
            ll = length(BerahasR{Idprob}.KKTD{DecayStep,sigma});
            if ll > 0 
                a = [];
                b = [];
                c = [];
                for lll = 1:ll 
                    a = [a; BerahasR{Idprob}.KKTD{DecayStep,sigma}{lll}(end)];
                    b = [b; BerahasR{Idprob}.TimeD{DecayStep,sigma}{lll}(end)];
                    c = [c; length(BerahasR{Idprob}.KKTD{DecayStep,sigma}{lll})];
                end
                if ~isnan(min(a))
                    KKTVec = [KKTVec; min(a)];
                    TimeVec = [TimeVec; min(b)];
                    CountVec = [CountVec; min(c)];
                end
            end 
        end
        BerhasDKKT{DecayStep,sigma} = KKTVec;
        BerhasDTime{DecayStep,sigma} = TimeVec;
        BerhasDCount{DecayStep,sigma} = CountVec;
    end 
end

%% Extract NonAdapSQP result 
NonAdapR = [NonAdapR1;NonAdapR2];
for ConStep = 1:4
    for sigma = 1:5 
        KKTVec = [];
        TimeVec = [];
        CountVec = [];
        for Idprob = 1:47
            ll = length(NonAdapR{Idprob}.KKTC{ConStep,sigma});
            if ll > 0 
                a = [];
                b = [];
                c = [];
                for lll = 1:ll 
                    a = [a; NonAdapR{Idprob}.KKTC{ConStep,sigma}{lll}(end)];
                    b = [b; NonAdapR{Idprob}.TimeC{ConStep,sigma}{lll}(end)];
                    c = [c; length(NonAdapR{Idprob}.KKTC{ConStep,sigma}{lll})];
                end
                if ~isnan(min(a))
                    KKTVec = [KKTVec; min(a)];
                    TimeVec = [TimeVec; min(b)];
                    CountVec = [CountVec; min(c)];
                end
            end 
        end
        NonAdapCKKT{ConStep,sigma} = KKTVec;
        NonAdapCTime{ConStep,sigma} = TimeVec;
        NonAdapCCount{ConStep,sigma} = CountVec;
    end 
end

for DecayStep = 1:2
    for sigma = 1:5 
        KKTVec = [];
        TimeVec = [];
        CountVec = [];
        for Idprob = 1:47
            ll = length(NonAdapR{Idprob}.KKTD{DecayStep,sigma});
            if ll > 0 
                a = [];
                b = [];
                c = [];
                for lll = 1:ll 
                    a = [a; NonAdapR{Idprob}.KKTD{DecayStep,sigma}{lll}(end)];
                    b = [b; NonAdapR{Idprob}.TimeD{DecayStep,sigma}{lll}(end)];
                    c = [c; length(NonAdapR{Idprob}.KKTD{DecayStep,sigma}{lll})];
                end
                if ~isnan(min(a))
                    KKTVec = [KKTVec; min(a)];
                    TimeVec = [TimeVec; min(b)];
                    CountVec = [CountVec; min(c)];
                end
            end 
        end
        NonAdapDKKT{DecayStep,sigma} = KKTVec;
        NonAdapDTime{DecayStep,sigma} = TimeVec;
        NonAdapDCount{DecayStep,sigma} = CountVec;
    end 
end


%% Plot
% Go over constant stepsize 
for ConStep = 1:4
    % Plot KKT residual
    data = cell(5, 3);
    for sigma = 1:size(data,1)
        Ac{sigma} = AdapKKT{sigma};
        Bc{sigma} = BerhasCKKT{ConStep, sigma};
        if length(NonAdapCKKT{ConStep, sigma})>0 
            Cc{sigma} = NonAdapCKKT{ConStep, sigma};
        else 
            Cc{sigma} = [NaN];
        end
    end
    data = vertcat(Ac,Bc,Cc);
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    col=[102,255,255, 200;
        51,153,255, 200;
        0, 0, 255, 200];
    col=col/255;
    multiple_boxplot(data',xlab,{'AdapSQP', 'L_1SQP', 'NonAdapSQP'},col',1)
%    filename = ['./Figure/KKTCon' num2str(ConStep) '.png'];
%    print('-dpng', filename)
    
    % Plot consuming time
    data = cell(5,3);
    for sigma = 1:size(data,1)
        Ac{sigma} = AdapTime{sigma};
        Bc{sigma} = BerhasCTime{ConStep, sigma};
        if length(NonAdapCTime{ConStep, sigma})>0 
            Cc{sigma} = NonAdapCTime{ConStep, sigma};
        else 
            Cc{sigma} = [NaN];
        end
    end
    data = vertcat(Ac,Bc,Cc);
    multiple_boxplot(data',xlab,{'AdapSQP', 'L_1SQP', 'NonAdapSQP'},col',2)
%    filename = ['./Figure/TimeCon' num2str(ConStep) '.png'];
%    print('-dpng', filename)
    
end

% Go over decay steps

for DecayStep = 1:2
    % Plot KKT residual
    data = cell(5, 3);
    for sigma = 1:size(data,1)
        Ac{sigma} = AdapKKT{sigma};
        Bc{sigma} = BerhasDKKT{DecayStep, sigma};
        Cc{sigma} = NonAdapDKKT{DecayStep, sigma};
    end
    data = vertcat(Ac,Bc,Cc);
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    col=[102,255,255, 200;
        51,153,255, 200;
        0, 0, 255, 200];
    col=col/255;
    multiple_boxplot(data',xlab,{'AdapSQP', 'L_1SQP', 'NonAdapSQP'},col',1)
    filename = ['./Figure/KKTDecay' num2str(DecayStep) '.png'];
    print('-dpng', filename)
    
    % Plot consuming time
    data = cell(5,3);
    for sigma = 1:size(data,1)
        Ac{sigma} = AdapTime{sigma};
        Bc{sigma} = BerhasDTime{DecayStep, sigma};
        Cc{sigma} = NonAdapDTime{DecayStep, sigma};
    end
    data = vertcat(Ac,Bc,Cc);
    multiple_boxplot(data',xlab,{'AdapSQP', 'L_1SQP', 'NonAdapSQP'},col',2)
    filename = ['./Figure/TimeDecay' num2str(DecayStep) '.png'];
    print('-dpng', filename)
    
end


%% Output convergence status
ProbId = fopen('./Parameter/problems.txt','r');
Prob = textscan(ProbId,'%s','delimiter','\n');

for sigma = 1:5 
    for Idprob = 1:47
        % result of AdapSQP
        la = length(AdapR{Idprob}.KKTStep{sigma});
        if la > 0
            a = [];
            for lll = 1:la 
                a = [a; AdapR{Idprob}.KKTStep{sigma}{lll}(end)];
            end
            mina = min(a);
            Ra = [1, log(mina)];
        else 
            Ra = [0, 100];
        end
        % result of BerahasSQP
        minb = NaN;
        for CStep = 1:4
            for lll = 1:length(BerahasR{Idprob}.KKTC{CStep, sigma})
                minb = min(minb, BerahasR{Idprob}.KKTC{CStep, sigma}{lll}(end));
            end 
        end
        for DStep = 1:2
            for lll = 1:length(BerahasR{Idprob}.KKTD{DStep, sigma})
                minb = min(minb, BerahasR{Idprob}.KKTD{DStep, sigma}{lll}(end));
            end 
        end
        if ~isnan(minb)
            Rb = [1, log(minb)];
        else 
            Rb = [0, 100];
        end
        
        % result of NonAdapSQP
        minc = NaN;
        for CStep = 1:4
            for lll = 1:length(NonAdapR{Idprob}.KKTC{CStep, sigma})
                minc = min(minc, NonAdapR{Idprob}.KKTC{CStep, sigma}{lll}(end));
            end 
        end
        for DStep = 1:2
            for lll = 1:length(NonAdapR{Idprob}.KKTD{DStep, sigma})
                minc = min(minc, NonAdapR{Idprob}.KKTD{DStep, sigma}{lll}(end));
            end 
        end
        if ~isnan(minc)
            Rc = [1, log(minc)];
        else 
            Rc = [0, 100];
        end

        filename = ['./Figure/sigma' num2str(sigma) '.txt'];
        if exist(filename, 'file') == 0
            f = fopen(filename, 'w+');
            fprintf(f, '%s', Prob{1}{Idprob});
            fprintf(f, ' [%3d, %8.4f]', Ra);
            fprintf(f, '[%9d, %15.4f]', Rb);
            fprintf(f, '[%16d, %20.4f]\n', Rc);
        else
            f = fopen(filename, 'a+');
            fprintf(f, '%s', Prob{1}{Idprob});
            fprintf(f, ' [%3d, %8.4f]', Ra);
            fprintf(f, '[%9d, %15.4f]', Rb);
            fprintf(f, '[%16d, %20.4f]\n', Rc);
        end
    end
end
            
    
       
    

    
    
    
    
     

        
            




