function [w b] = Perceptron(tr , label)

[d N] = size(tr);
% initialize weights and bias
w = zeros(d,1);
b = 0;
maxiter = 3;
for iter = 1 : maxiter
    for n = 1 : N
        % Check for the wrong decision
        if( (w'*tr(:,n) + b) * label(n) <= 0 )
            w = w + label(n) * tr(:,n);
            b = b + label(n);
        end
    end
end