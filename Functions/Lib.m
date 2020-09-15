function [p] = Lib(x,dx,ddx,t,polyorder,CrossedProducts,Algebraic,CoulombFriction)

if Algebraic==0
    
for i=1:polyorder
    px(:,i)=x.^i;
    pv(:,i)=dx.^i;
end


for j=1:polyorder
    for k=1:polyorder
        pxv(:,(j-1)*polyorder+k)=(x.^j).*(dx.^k);
    end
end

pCoulomb=sign(dx);


if CoulombFriction==0
    if CrossedProducts==1
        p=[ddx,px,pv,pxv];
    elseif CrossedProducts==0
        p=[ddx,px,pv];
    end
    
elseif CoulombFriction==1
    if CrossedProducts==1
        p=[ddx,px,pv,pxv,pCoulomb];
        
    elseif CrossedProducts==0
        p=[ddx,px,pv,pCoulomb];
    end    
end   

%%% Algebraic Data Library
elseif Algebraic==1
    pa(:,1)=Al(0,2,x,t)-4*Al(1,1,x,t)+2*Al(2,0,x,t);
    
    for i=1:polyorder
        px(:,i)=Al(2,2,x.^i,t);
    end
    
    pv(:,1)=Al(1,2,x,t)-2*Al(2,1,x,t);
    
    for j=2:polyorder
        pv(:,j)=Al(2,2,dx.^j,t);
    end
    
    for k=1:polyorder
        for l=1:polyorder
            pxv(:,(k-1)*polyorder+l)=Al(2,2,(x.^k).*(dx.^l),t);
        end
    end   
    
    pCoulomb=Al(2,2,sign(dx),t);
    
    if CoulombFriction==0
        if CrossedProducts==1
            p=[pa,px,pv,pxv]; 
        elseif CrossedProducts==0
            p=[pa,px,pv];
        end
        
    elseif CoulombFriction==1 
        if CrossedProducts==1
            p=[pa,px,pv,pxv]; 
        elseif CrossedProducts==0
            p=[pa,px,pv,pCoulomb];
        end
    end

end


    