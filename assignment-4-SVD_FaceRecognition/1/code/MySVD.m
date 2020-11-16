function [U, Sigma, V] = MySVD(A)
    [m,n] = size(A);
    Sigma = zeros(m,n);
    
    [U, valU] = eig(A*A');
    [~,idx] = sort(diag(valU));
    idx = fliplr(idx')';
    valU = valU(idx,idx);
    U = U(:,idx);
    
    [V, valV] = eig(A'*A);
    [~,idx] = sort(diag(valV));
    idx = fliplr(idx')';
    valV = valV(idx,idx);
    V = V(:,idx);
    
    if (m<n)
        Sigma(1:m,1:m) = sqrt(valU);
    else 
        Sigma(1:n,1:n) = sqrt(valV);
    end
        
    for i=1:min(m,n)
        if (dot(A*V(:,i),U(:,i))<0)
            V(:,i) = -V(:,i);
        end
    end

end
