function [U,Y,out]=gmf(A,Phi,r,lambda,options)
%gmf  graph contrainted low-rank matrix factorization.
% $$\min_{U'U=I,Y}||A-UY||_F^2+ \lambda tr(Y \Phi Y')$$
% options:
%   directmethod  ---{0|1(default)} ---use direct method or not
%   cholPsi       ---{0|1(default)} ---decompose Psi or not
%   maxit         --- max iteration number of PCG or symmlq
%   tol           ---{1e-5|small values} stop condition
[m,n]=size(A);
if ~exist('lambda','var'),lambda=0;end
if ~exist('options','var'),options=struct;end;
[tol,issvd,cholPsi,directmethod,maxit,display]=getoption(options,n);

%-------------direct method---------------
if directmethod && m<=5000&&n<=10000,
    if(lambda>0)
        Psi =lambda*Phi+speye(size(Phi));
        A_Psi = A/Psi;
    else
        if n<1000&&m>n,warning('gmf:svd','direct svds may be better');end
        A_Psi = A;
    end
    A_Psi_A = A_Psi*A';
    
    opts.issym=1;opts.isreal=1;
    opts.disp =0;opts.v0 = ones(size(A,1),1);
    A_Psi_A =(A_Psi_A+A_Psi_A');% symmetric is better    
    [U,~]=eigs(A_Psi_A,r,'LA',opts);
    Y=U'*A_Psi;
    
    out.E = norm(A,'fro')^2+norm(Y,'fro')^2 ...
        - 2*sum(sum(Y.*(U'*A)));
    if~isempty(Phi),out.E=out.E+lambda*sum(sum((Y*Phi).*Y));end
    out.algo='direct method';
    return;
end

% -------------iteration method--------------
if 0,  % initial guess
    idx = randperm(n);
    [U,~]=svd(full(A(:,idx(1:min(3*r,n)))),'econ');
    U = U(:,1:r);Y = U'*A;
else [U,~] = qr(full(A(:,1:r)),0);Y = U'*A; end

Psi =lambda*Phi+speye(size(Phi));
dPsi= spdiags(diag(Psi),0,n,n);
if cholPsi,Psi = chol(Psi,'lower');end;

err_=inf; maxiter=200; E=zeros(maxiter,1);nrmA2 =norm(A,'fro')^2;
for iter=1:maxiter
    err=updateY;
    updateU;
    
    E(iter)=err;
    if abs(err_-err)<err*tol,
        break;
    end;
    err_=err;
    if display,fprintf('iter:%d err:%5.5e\n',iter,err);end
end;

out.algo = 'iteraion method';
out.E=E(1:iter);
%  end of main

    function err=updateY
        B=U'*A;
        if cholPsi
            Y=B/(Psi')/Psi;
        else
            tol_=tol;%maxit=20;
            for i=1:r
                [y,~,~,iter_] = pcg(Psi,B(i,:).',tol_,maxit,dPsi,[],Y(i,:)');
                %[y,~,~,iter_] = symmlq(Psi,B(i,:).',tol_,maxit,dPsi,[],Y(i,:)');
                Y(i,:)=y';
                if i<5&&display,fprintf('%d.',iter_);end
            end
        end
        err=nrmA2+norm(Y,'fro')^2-2*sum(sum(Y.*B))+lambda*sum(sum((Y*Phi).*Y ));
    end % updateY

    function updateU
        % $$ \min_U ||A-UY||_F^2  s.t. U'U=I $$
        if issvd
            [U,~,G]=svd(A*Y','econ');
            U=U*G';
        else
            [U,~]=qr(A*Y',0);
        end
    end%  updateU

end
%-------------------end---------------------------

function [tol,issvd,cholPsi,directmethod,maxit,display] = getoption(options,n)
issvd = 1;
tol = 1e-5;
%  solve the linear system directly or not %
cholPsi = 0;if n<5000,cholPsi=1;end;
directmethod =1;
maxit = 20;
display =1;

if isfield(options,'issvd'),issvd = options.issvd;end
if isfield(options,'tol'),tol = options.tol;end
if isfield(options,'cholPsi'),cholPsi = options.cholPsi;end
if isfield(options,'directmethod'),directmethod = options.directmethod;end
if isfield(options,'maxit'),maxit = options.maxit;end
if isfield(options,'display'),display=options.display;end
end