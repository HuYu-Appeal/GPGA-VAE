function homo()
    % 定义材料参数
    lx = 1; % 晶胞在 x 方向的尺寸
    ly = 1; % 晶胞在 y 方向的尺寸
    lz = 1; % 晶胞在 z 方向的尺寸
    lambda = 115.4; % 第一拉梅参数 (GPa)
    mu = 76.9; % 第二拉梅参数 (GPa)
    
    % 读取 HDF5 文件中的 voxel 模型
    voxel = readVoxelModel('voxel_models.h5');
    
    % 调用 homo3D 函数计算均匀化弹性张量
    CH = homo3D(lx, ly, lz, lambda, mu, voxel);
    
    % 输出结果
    disp('均匀化弹性张量 (6x6 矩阵):');
    disp(CH);
end

function voxel = readVoxelModel(filename)
    % 从 HDF5 文件中读取 voxel 模型
    % 获取文件中的所有键
    info = h5info(filename);
    if isempty(info)
        error('HDF5 文件为空或不存在');
    end
    
    % 选择第一个键下的 voxel 数据集
    firstKey = info.GroupHierarchy.Groups(1).Name;
    voxelPath = [firstKey '/voxel'];
    
    % 从 HDF5 文件中读取 voxel 模型
    voxel = h5read(filename, voxelPath);
    % 确保 voxel 是一个逻辑矩阵（0 或 1）
    voxel = logical(voxel);
end

function CH = homo3D(lx, ly, lz, lambda, mu, voxel)
    % 初始化
    [nelx, nely, nelz] = size(voxel); % 获取晶胞的网格尺寸
    dx = lx / nelx; dy = ly / nely; dz = lz / nelz; % 单元尺寸
    nel = nelx * nely * nelz; % 单元总数
    [keLambda, keMu, feLambda, feMu] = hexahedron(dx / 2, dy / 2, dz / 2); % 单元刚度矩阵和载荷向量
    
    % 节点编号和自由度
    nodenrs = reshape(1:(1 + nelx) * (1 + nely) * (1 + nelz), 1 + nelx, 1 + nely, 1 + nelz);
    edofVec = reshape(3 * nodenrs(1:end-1, 1:end-1, 1:end-1) + 1, nel, 1);
    addx = [0 1 2 3 * nelx + [3 4 5 0 1 2] -3 -2 -1];
    addxy = 3 * (nely + 1) * (nelx + 1) + addx;
    edof = repmat(edofVec, 1, 24) + repmat([addx addxy], nel, 1);
    
    % 周期性边界条件
    nn = (nelx + 1) * (nely + 1) * (nelz + 1); % 总节点数
    nnP = nelx * nely * nelz; % 独特节点数
    nnPArray = reshape(1:nnP, nelx, nely, nelz);
    nnPArray(end + 1, :, :) = nnPArray(1, :, :);
    nnPArray(:, end + 1, :) = nnPArray(:, 1, :);
    nnPArray(:, :, end + 1) = nnPArray(:, :, 1);
    dofVector = zeros(3 * nn, 1);
    dofVector(1:3:end) = 3 * nnPArray(:) - 2;
    dofVector(2:3:end) = 3 * nnPArray(:) - 1;
    dofVector(3:3:end) = 3 * nnPArray(:);
    edof = dofVector(edof);
    ndof = 3 * nnP;
    
    % 组装全局刚度矩阵和载荷向量
    iK = kron(edof, ones(24, 1))';
    jK = kron(edof, ones(1, 24))';
    lambda = lambda * (voxel == 1);
    mu = mu * (voxel == 1);
    sK = keLambda(:) * lambda(:).' + keMu(:) * mu(:).';
    K = sparse(iK(:), jK(:), sK(:), ndof, ndof);
    K = 1/2 * (K + K');
    iF = repmat(edof', 6, 1);
    jF = [ones(24, nel); 2 * ones(24, nel); 3 * ones(24, nel);...
        4 * ones(24, nel); 5 * ones(24, nel); 6 * ones(24, nel);];
    sF = feLambda(:) * lambda(:).' + feMu(:) * mu(:).';
    F = sparse(iF(:), jF(:), sF(:), ndof, 6);
    
    % 求解全局方程
    activedofs = edof(voxel == 1, :); activedofs = sort(unique(activedofs(:)));
    X = zeros(ndof, 6);
    L = ichol(K(activedofs(4:end), activedofs(4:end)));
    for i = 1:6
        X(activedofs(4:end), i) = pcg(K(activedofs(4:end),...
            activedofs(4:end)), F(activedofs(4:end), i), 1e-10, 300, L, L');
    end
    
    % 计算均匀化弹性张量
    X0 = zeros(nel, 24, 6);
    X0_e = zeros(24, 6);
    ke = keMu + keLambda;
    fe = feMu + feLambda;
    X0_e([4 7:11 13:24], :) = ke([4 7:11 13:24], [4 7:11 13:24])...
                               \fe([4 7:11 13:24], :);
    for i = 1:6
        X0(:, :, i) = kron(X0_e(:, i)', ones(nel, 1));
    end
    CH = zeros(6);
    volume = lx * ly * lz;
    for i = 1:6
        for j = 1:6
            sum_L = ((X0(:, :, i) - X(edof + (i - 1) * ndof)) * keLambda) .*...
                (X0(:, :, j) - X(edof + (j - 1) * ndof));
            sum_M = ((X0(:, :, i) - X(edof + (i - 1) * ndof)) * keMu) .*...
                (X0(:, :, j) - X(edof + (j - 1) * ndof));
            sum_L = reshape(sum(sum_L, 2), nelx, nely, nelz);
            sum_M = reshape(sum(sum_M, 2), nelx, nely, nelz);
            CH(i, j) = 1 / volume * sum(sum(sum(lambda .* sum_L + mu .* sum_M)));
        end
    end
end

function [keLambda, keMu, feLambda, feMu] = hexahedron(a, b, c)
    % Constitutive matrix contributions
    CMu = diag([2 2 2 1 1 1]); CLambda = zeros(6); CLambda(1:3, 1:3) = 1;
    % Three Gauss points in both directions
    xx = [-sqrt(3 / 5), 0, sqrt(3 / 5)]; yy = xx; zz = xx;
    ww = [5 / 9, 8 / 9, 5 / 9];
    % Initialize
    keLambda = zeros(24, 24); keMu = zeros(24, 24);
    feLambda = zeros(24, 6); feMu = zeros(24, 6);
    for ii = 1:length(xx)
        for jj = 1:length(yy)
            for kk = 1:length(zz)
                % Integration point
                x = xx(ii); y = yy(jj); z = zz(kk);
                % Stress strain displacement matrix
                qx = [ -((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,...
                    ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,...
                    ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8];
                qy = [ -((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,...
                    ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,...
                    ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8];
                qz = [ -((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,...
                    ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,...
                    ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8];
                % Jacobian
                J = [qx; qy; qz] * [-a a a -a -a a a -a; -b -b b b -b -b b b;...
                    -c -c -c -c c c c c]';
                qxyz = J \ [qx; qy; qz];
                B_e = zeros(6, 3, 8);
                for i_B = 1:8
                    B_e(:, :, i_B) = [qxyz(1, i_B) 0 0;
                                      0 qxyz(2, i_B) 0;
                                      0 0 qxyz(3, i_B);
                                      qxyz(2, i_B) qxyz(1, i_B) 0;
                                      0 qxyz(3, i_B) qxyz(2, i_B);
                                      qxyz(3, i_B) 0 qxyz(1, i_B)];
                end
                B = [B_e(:, :, 1) B_e(:, :, 2) B_e(:, :, 3) B_e(:, :, 4) B_e(:, :, 5)...
                    B_e(:, :, 6) B_e(:, :, 7) B_e(:, :, 8)];
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