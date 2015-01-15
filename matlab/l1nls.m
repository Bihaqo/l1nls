function [x, val] = l1nls(A, b)
	% Solves an optimization problem
	% min. ||x||_1
	% s.t.
	% A x = b
	%
	% where ||x||_1 is the 1 norm of the vector x.
	%

	if (~isvector(b))
		error('b must be a vector.')
	end
	[m, n] = size(A);
	if (m ~= length(b))
		error('The length of the vector b must match the number of rows in the matrix A.');
	end

	Anew = [A, -A];
	f = ones(2 * n, 1);
	lb = zeros(2 * n, 1);
	[auxiliary, val] = linprog(f, [], [], Anew, b, lb);
	x = auxiliary(1:n) - auxiliary(n+1:end);
end
