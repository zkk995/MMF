function [W,L,Ni]=laplacian(R,p,k,isone)

if ~exist('k','var'),k=p;end;

n=size(R,2);
Ni=zeros(k,n);
Vi=zeros(p,n);


Rsqure=sum(R.*R,1)/2;

if ((n<5000)),
    Dist = repmat(Rsqure,n,1) - R'*R+repmat(Rsqure(:),1,n);
    [V,idx]=sort(Dist,'ascend');
    Ni=idx((1:k)+1,:);
    Vi=V((1:p)+1,:);
    sigma2 =mean(Dist(:,end));clear Dist;
else
    block=round(1500^2/n);i=1:block;
    for nb=block:block:n
        dist=repmat(Rsqure,block,1)-R(:,i)'*R;
        [V,idx]=sort(dist,2,'ascend');
        Ni(:,i)=idx(:,(1:k)+1)';
        Vi(:,i)=V(:,(1:p)+1)'+repmat(Rsqure(i),p,1);
        i=block+i;
    end;
    i=i(1):n;
    dist=repmat(Rsqure,length(i),1)-R(:,i)'*R;
    [V,idx]=sort(dist,2,'ascend');
    Ni(:,i)=idx(:,(1:k)+1)';
    Vi(:,i)=V(:,(1:p)+1)'+repmat(Rsqure(i),p,1);
    sigma2 =mean(dist(end,:)+Rsqure(end));
end


Ir=Ni(1:p,:);Jc=ones(p,1)*(1:n);

if ~exist('isone','var'),isone=0;end
if isone
    W = sparse(Ir(:),Jc(:),ones(p*n,1),n,n);
else
    W = sparse(Ir(:),Jc(:),exp(-Vi(:)./(sigma2*4)),n,n);
end

W=max(W,W'); clear Ir Jc;

L=graph2Phi;

    function L=graph2Phi
        
        DCol = full(sum(W,2));
        D = spdiags(DCol,0,speye(size(W,1)));
        L = D - W;
    end
end


%%

