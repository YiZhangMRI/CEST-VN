function [tumor_det] = tumor_detection(Input,col)
% tumor_detection: Determine the location of the tumor
[r,c]   = size(Input);
Input   =double(Input);
Length = (r*c); 
Dataset = reshape(Input,[Length,1]);
Clusters=5; % k CLUSTERS
Cluster1=zeros(Length,1);
Cluster2=zeros(Length,1);
Cluster3=zeros(Length,1);
Cluster4=zeros(Length,1);
Cluster5=zeros(Length,1);
miniv = min(min(Input));
maxiv = max(max(Input));
range = maxiv - miniv;
stepv = range/Clusters;
incrval = stepv;
for i = 1:Clusters
    K(i).centroid = incrval;
    incrval = incrval + stepv;
end

update1=0;
update2=0;
update3=0;
update4=0;
update5=0;

mean1=2;
mean2=2;
mean3=2;
mean4=2;
mean5=2;

while  ((mean1 ~= update1) && (mean2 ~= update2) && (mean3 ~= update3) && (mean4 ~= update4) && (mean5 ~= update5))

    mean1=K(1).centroid;
    mean2=K(2).centroid;
    mean3=K(3).centroid;
    mean4=K(4).centroid;
    mean5=K(5).centroid;

    for i=1:Length
        for j = 1:Clusters
            temp= Dataset(i);
            difference(j) = abs(temp-K(j).centroid);
        end

        [y,ind]=min(difference);

        if      ind==1
                Cluster1(i)   =temp;
        elseif  ind==2
                Cluster2(i)   =temp;
        elseif  ind==3
                Cluster3(i)   =temp;
        elseif  ind==4
                Cluster4(i)   =temp;
        elseif  ind==5
                Cluster5(i)   =temp;
        end
    end

    %UPDATE CENTROIDS
    cout1=0;
    cout2=0;
    cout3=0;
    cout4=0;
    cout5=0;

    for i=1:Length
        Load1=Cluster1(i);
        Load2=Cluster2(i);
        Load3=Cluster3(i);
        Load4=Cluster4(i);
        Load5=Cluster5(i);

        if Load1 ~= 0
            cout1=cout1+1;
        end

        if Load2 ~= 0
            cout2=cout2+1;
        end
    %     
        if Load3 ~= 0
            cout3=cout3+1;
        end

        if Load4 ~= 0
            cout4=cout4+1;
        end

        if Load5 ~= 0
            cout5=cout5+1;
        end
    end

    Mean_Cluster(1)=sum(Cluster1)/cout1;
    Mean_Cluster(2)=sum(Cluster2)/cout2;
    Mean_Cluster(3)=sum(Cluster3)/cout3;
    Mean_Cluster(4)=sum(Cluster4)/cout4;
    Mean_Cluster(5)=sum(Cluster5)/cout5;
    %reload
    for i = 1:Clusters
        K(i).centroid = Mean_Cluster(i);
    end

    update1=K(1).centroid;
    update2=K(2).centroid;
    update3=K(3).centroid;                                  
    update4=K(4).centroid;
    update5=K(5).centroid;
end

tumor_det =reshape(Cluster5,[col col]);
end

