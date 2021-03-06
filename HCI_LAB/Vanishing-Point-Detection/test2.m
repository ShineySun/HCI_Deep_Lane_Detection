numPointRank = 100; % top N candidate points
camera.K = [(640)/2, 0, 320/2; 0, (640)/2, 320/2; 0, 0, 1];

VD3D(1:3,1) = ([-0.17256396, -0.04193904,  0.98410507]);
VD3D(1:3,2) = ([0.8503485 , 0.51822425, 0.09138414]);
VD3D(1:3,3) = ([-0.72922609, -0.40467232,  0.55178766]);
VD3D(1:3,4) = ([-0.17327214, -0.04237485,  0.98396196]);
VD3D(1:3,5) = ([-0.1734666 , -0.04249453,  0.98392253]);
VD3D(1:3,6) = ([-0.83354174, -0.48573678,  0.26318806]);
VD3D(1:3,7) = ([-0.17169292, -0.04140307,  0.98428011]);
VD3D(1:3,8) = ([-1.05238828e-01, -7.13483583e-04,  9.94446721e-01]);
VD3D(1:3,9) = ([-0.16822587, -0.03357938,  0.98517637]);
VD3D(1:3,10) = ([-0.16771045, -0.03258787,  0.98529754]);
VD3D(1:3,11) = ([-0.17266684, -0.04213761,  0.98407855]);
VD3D(1:3,12) = ([-0.17732262, -0.05113937,  0.98282321]);
VD3D(1:3,13) = ([-0.1648435 , -0.02707945,  0.98594793]);
VD3D(1:3,14) = ([-0.17172769, -0.04032549,  0.98431878]);
VD3D(1:3,15) = ([-0.16216609, -0.02194536,  0.98651942]);
VD3D(1:3,16) = ([-0.26860511, -0.095847  ,  0.95846995]);
VD3D(1:3,17) = ([-0.19859846, -0.05231107,  0.97868391]);
VD3D(1:3,18) = ([-0.16995145, -0.03464113,  0.98484339]);
VD3D(1:3,19) = ([-0.72338509, -0.39667314,  0.56512338]);
VD3D(1:3,20) = ([-0.17187084, -0.03582248,  0.98446796]);
VD3D(1:3,21) = ([-0.12914883, -0.00961346,  0.99157862]);
VD3D(1:3,22) = ([-0.20062844, -0.05310834,  0.97822683]);
VD3D(1:3,23) = ([-0.16954117, -0.03372618,  0.98494585]);
VD3D(1:3,24) = ([0.73736488, 0.48185495, 0.47340135]);
VD3D(1:3,25) = ([-0.17189074, -0.03518761,  0.98448738]);
VD3D(1:3,26) = ([-0.13291688, -0.01101835,  0.99106594]);
VD3D(1:3,27) = ([-0.17344306, -0.04244184,  0.98392896]);
VD3D(1:3,28) = ([-0.21698747, -0.059538  ,  0.97435705]);
VD3D(1:3,29) = ([-0.17168163, -0.0417515 ,  0.98426736]);
VD3D(1:3,30) = ([0.54325812, 0.23055705, 0.80728809]);
VD3D(1:3,31) = ([-0.1672175 , -0.02855084,  0.98550655]);
VD3D(1:3,32) = ([-0.1717791 , -0.03872116,  0.98437422]);
VD3D(1:3,33) = ([-0.1646661 , -0.02288111,  0.98608394]);
VD3D(1:3,34) = ([-0.17200435, -0.03151953,  0.98459181]);
VD3D(1:3,35) = ([-0.14796404, -0.01663491,  0.98885283]);
VD3D(1:3,36) = ([-0.17217994, -0.02569525,  0.98473033]);


% change (candidate) vanishing directions (3-d) to (candidate) vanishing points (2-d)
for i=1:size(VD3D,2),
    VP(:,i) = camera.K*[VD3D(1,i);VD3D(2,i);VD3D(3,i)];
    VP(:,i) = VP(:,i)./VP(3,i);
end

% make score for each (candidate) vanishing points
score_matrix = zeros(size(VD3D,2),size(VD3D,2));
for k = 1:size(VD3D,2),
    for l = k+1:size(VD3D,2),
        dist = norm([VP(1,k), VP(2,k)] - [VP(1,l), VP(2,l)]);
        if dist<1,
            dist = 1;
        end
        score_matrix(k,l) = 1/dist;
        score_matrix(l,k) = score_matrix(k,l);
    end
end


score_vector = sum(score_matrix)/sum(sum(score_matrix));
%score_vector = sum(score_matrix);
%score_vector2 = sum(score_vector)
size(score_vector);
%display(score_vector)

[~, score_idx] = sort(score_vector, 'descend');


% extract top N (candidate) vanishing points
if length(VP) < numPointRank,
    numPointRank = length(VP);
end
for i=1:numPointRank,
    %display(score_idx(i))
    topN_VP(:,i) = VP(:,score_idx(i));
end

%display(topN_VP)

vp_cand = topN_VP(1:2,:);
%size(vp_cand)
%disp(vp_cand)

[fClusterPCoordinate, fClusterPVoting,fClusterPSumweight,fPointToClusterNo] = clustering(vp_cand);
