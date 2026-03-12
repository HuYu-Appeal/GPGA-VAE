function CH = homo3D(lx, ly, lz, lambda, mu, voxel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lx       = Unit cell length in x-direction.
% ly       = Unit cell length in y-direction.
% lz       = Unit cell length in z-direction.
% lambda   = Lame's first parameter for solid materials (scalar).
% mu       = Lame's second parameter for solid materials (scalar).
% voxel    = Material indicator matrix. Used to determine nelx/nely/nelz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% INITIALIZE
[nelx, nely, nelz] = size(voxel); % Size of voxel model along x, y, and z axis

% 保存原始标量参数（必须在任何数组操作前保存！）
lambda_original = lambda; % 原始输入的Lame参数（标量）
mu_original = mu;         % 原始输入的Lame参数（标量）

% 材料过滤（将标量lambda/mu扩展为与voxel同尺寸的数组）
lambda_filtered = lambda_original .* (voxel == 1);  % 此时lambda_filtered变为三维数组
mu_filtered = mu_original .* (voxel == 1);          % 此时mu_filtered变为三维数组

% Stiffness matrix
dx = lx / nelx; 
dy = ly / nely; 
dz = lz / nelz;
nel = nelx * nely * nelz;

[keLambda, keMu, feLambda, feMu] = hexahedron(dx / 2, dy / 2, dz / 2);

% Node numbers and element degrees of freedom for full (not periodic) mesh
nodenrs = reshape(1:(1+nelx)*(1+nely)*(1+nelz), 1+nelx, 1+nely, 1+nelz);
edofVec = reshape(3 * nodenrs(1:end-1, 1:end-1, 1:end-1) + 1, nel, 1);
addx = [0 1 2 3*nelx + [3 4 5 0 1 2] -3 -2 -1];
addxy = 3 * (nely + 1) * (nelx + 1) + addx;
edof = repmat(edofVec, 1, 24) + repmat([addx addxy], nel, 1);

%% IMPOSE PERIODIC BOUNDARY CONDITIONS
% Use original edofMat to index into list with the periodic dofs
nn = (nelx + 1) * (nely + 1) * (nelz + 1); % Total number of nodes
nnP = (nelx) * (nely) * (nelz);    % Total number of unique nodes
nnPArray = reshape(1:nnP, nelx, nely, nelz);
% Extend with a mirror of the back border
nnPArray(end + 1, :, :) = nnPArray(1, :, :);
% Extend with a mirror of the left border
nnPArray(:, end + 1, :) = nnPArray(:, 1, :);
% Extend with a mirror of the top border
nnPArray(:, :, end + 1) = nnPArray(:, :, 1);
% Make a vector into which we can index using edofMat:
dofVector = zeros(3 * nn, 1);
dofVector(1:3:end) = 3 * nnPArray(:) - 2;
dofVector(2:3:end) = 3 * nnPArray(:) - 1;
dofVector(3:3:end) = 3 * nnPArray(:);
edof = dofVector(edof);
ndof = 3 * nnP;

%% ASSEMBLE GLOBAL STIFFNESS MATRIX AND LOAD VECTORS
% Indexing vectors
iK = kron(edof, ones(24, 1))';
jK = kron(edof, ones(1, 24))';
% Material properties assigned to voxels with materials
lambdaVoxel = lambda_filtered;  
muVoxel = mu_filtered;
% The corresponding stiffness matrix entries
sK = keLambda(:) * lambdaVoxel(:).' + keMu(:) * muVoxel(:).';
K = sparse(iK(:), jK(:), sK(:), ndof, ndof);
K = 1/2 * (K + K');

% Assembly three load cases corresponding to the three strain cases
iF = repmat(edof', 6, 1);
jF = [ones(24, nel); 2 * ones(24, nel); 3 * ones(24, nel);...
    4 * ones(24, nel); 5 * ones(24, nel); 6 * ones(24, nel);];
sF = feLambda(:) * lambdaVoxel(:).' + feMu(:) * muVoxel(:).';
F  = sparse(iF(:), jF(:), sF(:), ndof, 6);

%% SOLUTION
% solve by PCG method, remember to constrain one node
activedofs = edof(voxel == 1, :); 
activedofs = sort(unique(activedofs(:)));
X = zeros(ndof, 6);
% 修改PCG部分（增加偏移补偿和容差）
opts.type = 'ict';
opts.droptol = 1e-4;       % 更宽松的丢弃阈值
opts.diagcomp = 0.5;       % 增大偏移量（针对高条件数）
L = ichol(K(activedofs(4:end), activedofs(4:end)), opts);
tolerance = 1e-12;         % 更严格的收敛标准
max_iterations = 2000;     % 增加迭代次数
for i = 1:6
    [X(activedofs(4:end), i), flag, relres, iter] = pcg(K(activedofs(4:end),...
        activedofs(4:end)), F(activedofs(4:end), i), tolerance, max_iterations, L, L');
    if flag ~= 0
        disp(['PCG did not converge for load case ', num2str(i), ...
            '. Relative residual: ', num2str(relres), ...
            ', Iterations: ', num2str(iter)]);
    end
end

%% HOMOGENIZATION
% The displacement vectors corresponding to the unit strain cases
X0 = zeros(nel, 24, 6);
% The element displacements for the six unit strains
X0_e = zeros(24, 6);
% Fix degrees of nodes [1 2 3 5 6 12];
ke = keMu + keLambda; % Here the exact ratio does not matter, because
fe = feMu + feLambda; % it is reflected in the load vector
X0_e([4 7:11 13:24], :) = ke([4 7:11 13:24], [4 7:11 13:24]) \ fe([4 7:11 13:24], :);
X0(:,:,1) = kron(X0_e(:, 1)', ones(nel, 1)); % epsilon0_11 = (1,0,0,0,0,0)
X0(:,:,2) = kron(X0_e(:, 2)', ones(nel, 1)); % epsilon0_22 = (0,1,0,0,0,0)
X0(:,:,3) = kron(X0_e(:, 3)', ones(nel, 1)); % epsilon0_33 = (0,0,1,0,0,0)
X0(:,:,4) = kron(X0_e(:, 4)', ones(nel, 1)); % epsilon0_12 = (0,0,0,1,0,0)
X0(:,:,5) = kron(X0_e(:, 5)', ones(nel, 1)); % epsilon0_23 = (0,0,0,0,1,0)
X0(:,:,6) = kron(X0_e(:, 6)', ones(nel, 1)); % epsilon0_13 = (0,0,0,0,0,1)
CH = zeros(6);
volume = lx * ly * lz;
for i = 1:6
    for j = 1:6
        sum_L = ((X0(:,:,i) - X(edof + (i-1)*ndof)) * keLambda) .* ...
            (X0(:,:,j) - X(edof + (j-1)*ndof));
        sum_M = ((X0(:,:,i) - X(edof + (j-1)*ndof)) * keMu) .* ...
            (X0(:,:,j) - X(edof + (j-1)*ndof));
        sum_L = reshape(sum(sum_L, 2), nelx, nely, nelz);
        sum_M = reshape(sum(sum_M, 2), nelx, nely, nelz);
        % Homogenized elasticity tensor
        CH(i, j) = 1/volume * sum(sum(sum(lambdaVoxel .* sum_L + muVoxel .* sum_M)));
    end
end

%% 强制对称性和立方对称性
CH = (CH + CH') / 2;                    % 全局对称
CH(1:3,4:6) = 0; CH(4:6,1:3) = 0;      % 清零耦合项（立方对称要求）
% 原错误代码
% CH(4:6,4:6) = diag(mean(diag(CH(4:6,4:6))) * eye(3);

% 修正后的代码
CH(4:6,4:6) = diag(repmat(mean(diag(CH(4:6,4:6))), 3, 1));

%% 添加模量计算部分（在函数末尾）
% 计算基材模量时使用原始标量值
nu = lambda_original / (2 * (lambda_original + mu_original)); % 泊松比
Es = 2 * mu_original * (1 + nu);              % 基材杨氏模量
Gs = mu_original;                             % 基材剪切模量

% 验证变量类型
disp(['lambda_original is scalar: ', num2str(isscalar(lambda_original))]);
disp(['mu_original is scalar: ', num2str(isscalar(mu_original))]);

% 计算柔度矩阵时处理可能的奇异矩阵
try
    S = inv(CH);
catch ME
    warning(ME.identifier, '%s', ME.message);
    S = zeros(6);
end

% 提取有效模量
E1 = 1 / S(1,1);    % X方向杨氏模量
E2 = 1 / S(2,2);    % Y方向杨氏模量
E3 = 1 / S(3,3);    % Z方向杨氏模量
G12 = 1 / (2 * S(4,4)); % XY平面剪切模量
G23 = 1 / (2 * S(5,5)); % YZ平面剪切模量
G13 = 1 / (2 * S(6,6)); % XZ平面剪切模量

% 返回结果
CH = struct('CH', CH, 'E1', E1, 'E2', E2, 'E3', E3, 'G12', G12, 'G23', G23, 'G13', G13);

end

%% COMPUTE ELEMENT STIFFNESS MATRIX AND LOAD VECTOR
function [keLambda, keMu, feLambda, feMu] = hexahedron(a, b, c)
% Constitutive matrix contributions
CMu = diag([2 2 2 1 1 1]); 
CLambda = zeros(6); 
CLambda(1:3,1:3) = 1;

% Three Gauss points in both directions
xx = [-sqrt(3/5), 0, sqrt(3/5)]; 
yy = xx; 
zz = xx;
ww = [5/9, 8/9, 5/9];

% Initialize
keLambda = zeros(24,24); 
keMu = zeros(24,24);
feLambda = zeros(24,6); 
feMu = zeros(24,6);

for ii = 1:length(xx)
    for jj = 1:length(yy)
        for kk = 1:length(zz)
            % Integration point
            x = xx(ii); 
            y = yy(jj); 
            z = zz(kk);
            
            % Stress-strain-displacement matrix
            qx = [ -((y-1)*(z-1))/8, ((y-1)*(z-1))/8, -((y+1)*(z-1))/8,...
                ((y+1)*(z-1))/8, ((y-1)*(z+1))/8, -((y-1)*(z+1))/8,...
                ((y+1)*(z+1))/8, -((y+1)*(z+1))/8];
            qy = [ -((x-1)*(z-1))/8, ((x+1)*(z-1))/8, -((x+1)*(z-1))/8,...
                ((x-1)*(z-1))/8, ((x-1)*(z+1))/8, -((x+1)*(z+1))/8,...
                ((x+1)*(z+1))/8, -((x-1)*(z+1))/8];
            qz = [ -((x-1)*(y-1))/8, ((x+1)*(y-1))/8, -((x+1)*(y+1))/8,...
                ((x-1)*(y+1))/8, ((x-1)*(y-1))/8, -((x+1)*(y-1))/8,...
                ((x+1)*(y+1))/8, -((x-1)*(y+1))/8];
            
            % Jacobian
            J = [qx; qy; qz] * [-a a a -a -a a a -a; -b -b b b -b -b b b;...
                -c -c -c -c c c c c]';
            qxyz = J \ [qx; qy; qz];
            
            B_e = zeros(6,3,8);
            for i_B = 1:8
                B_e(:,:,i_B) = [qxyz(1,i_B)   0             0;
                                0             qxyz(2,i_B)   0;
                                0             0             qxyz(3,i_B);
                                qxyz(2,i_B)   qxyz(1,i_B)   0;
                                0             qxyz(3,i_B)   qxyz(2,i_B);
                                qxyz(3,i_B)   0             qxyz(1,i_B)];
            end
            
            B = [B_e(:,:,1) B_e(:,:,2) B_e(:,:,3) B_e(:,:,4) B_e(:,:,5)...
                B_e(:,:,6) B_e(:,:,7) B_e(:,:,8)];
            
            % Weight factor at this point
            weight = det(J) * ww(ii) * ww(jj) * ww(kk);
            
            % Element matrices
            keLambda = keLambda + weight * B' * CLambda * B;
            keMu = keMu + weight * B' * CMu * B;
            % Element loads
            feLambda = feLambda + weight * B' * CLambda;       
            feMu = feMu + weight * B' * CMu; 
        end
    end
end
end



