function multiple_boxplot(data,xlab,Mlab,col,Id,IdLeg)

if ~iscell(data)
    error('Input data is not even a cell array!');
end

% Get sizes
L=size(data,1);
M=size(data,2);

% Calculate the positions of the boxes
range=(M+2)*L;
positions=1+((1:range)-1)*0.25;
positions([(0:L-1)*(M+2)+(M+1) (0:L-1)*(M+2)+(M+2)]) = [];

% Extract data and label it in the group correctly
x=[]; group=[]; Len = [];
for ii=1:L
    for jj=1:M
        aux=data{ii,jj};
        if Id < 2
            aux(aux>10^5)=[];
        end
        x=vertcat(x,aux(:));
        group=vertcat(group,ones(size(aux(:)))*jj+(ii-1)*M);
        if isnan(aux)
            Len = [Len; 0];
        else
            Len = [Len; length(aux)];
        end        
    end
end

% Plot it
boxplot(x,group, 'positions', positions,'symbol', '');

% Set the Xlabels
aux=reshape(positions,M,[]);
labelpos = sum(aux,1)./M;
set(gca,'xtick',labelpos)
set(gca,'YScale','log')
xlabel('$\sigma^2$','interpreter','latex','FontSize',40)
set(gca,'xticklabel',xlab,'FontSize',20);
color=repmat(col, 1, L);
% Apply colors
h = findobj(gca,'Tag','Box');
for jj=1:length(h)
   p(jj) =patch(get(h(jj),'XData'),get(h(jj),'YData'),color(1:3,jj)','FaceAlpha',color(4,jj));
end
p_sub =[];
for i = 1:M
    p_sub = [p_sub, p(i)];
end


if Id == 1
    ylabel('KKT Residual','FontSize',30)
    ylim([10^-6, 10^3])
%    for i = 1:M*L
%        text(positions(i), 10^2, sprintf('%d',Len(i)), 'FontSize',13);
%    end
elseif Id == 1.5 
    ylabel('KKT Residual','FontSize',30)
    ylim([10^-6, 10^-2])    
%    for i = 1:M*L
%        text(positions(i), 10^-1, sprintf('%d',Len(i)), 'FontSize',13);
%    end
elseif Id == 2
    ylabel('# Grad Samples','FontSize',30)
else
    ylabel('# Obj Samples','FontSize',30)
    ylim([10^5, 10^9])    
end

if IdLeg == 1
    legend(flip(p_sub),(Mlab),'Orientation','horizontal','Interpreter','latex');
end

end
   