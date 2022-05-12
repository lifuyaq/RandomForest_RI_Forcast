clc;
clear;
load('/home/lfy/Desktop/personal_fudan/senior_1/final_project/data/factor.mat');
vmax1=factor(:,5);
vmax2=factor(:,5);
% % % % % % %%%%%%%%%Vmax-IR%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
IR1=factor(:,7);
IR2=factor(:,7);
Vmax=factor(IR1>0&factor(:,2)>1,5);
IR1=IR1(IR1>0&factor(:,2)>1);
a=sort(Vmax);
b=a;
for i=1:length(a)-1
    if a(i)==a(i+1)
        b(i+1)=0;
    end
end
c=b(find(b~=0));
N95=[];
N50=[];
for j=1:length(c)
    t1=find(Vmax==c(j));
    d=IR1(t1);
   e=sort(d);
   n95=prctile(e,95);
   n50=prctile(e,50);
   N95=[N95
       n95];
   N50=[N50
       n50];
end
scatter(Vmax,IR1,3,[0.5 0.5 0.5],'filled');
hold on;
m=polyfit(c,N95,2);
d=polyval(m,c,1);
n=polyfit(c,N50,2);
e=polyval(n,c,1);
h1=plot(c(2:end-1),d(2:end-1),'r','LineWidth',1);
hold on;
h2=plot(c(2:end-1),e(2:end-1),'b','LineWidth',1);