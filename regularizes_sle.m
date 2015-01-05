function [x, val] = regularizes_sle(A, b)
	% Solves an optimization problem
	% min. ||x||_1
	% s.t.
	% A x = b
	% 
	% where ||x||_1 is the 1 norm of the vector x.
	% 

	[m, n] = size(A);
	Anew = [A, -A];
	f = ones(2 * n, 1);
	lb = zeros(2 * n, 1);
	[auxiliary, val] = linprog(f, [], [], Anew, b, lb);
	x = auxiliary(1:n) - auxiliary(n+1:end);
end