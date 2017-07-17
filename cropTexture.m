clearvars;
clc;
height=512;
width=512;
patchHeight=448;
patchWidth=448;
stepX=8;
stepY=8;
resizedH=299;
resizedW=299;
if isdir('Texcrop')
    rmdir('Texcrop','s');
end
mkdir('Texcrop');
for i=1:450
    fileName=[num2str(i) '.png'];
    image=imread(['rendedtextures/' fileName]);
    k=1;
    for m=1:stepX:height-patchHeight+1
        for n=1:stepY:width-patchWidth+1
            imwrite(imresize(image(m:m+patchHeight-1,n:n+patchWidth-1,1),[resizedH,resizedW]),['Texcrop/' num2str(i) '_' num2str(k) '.png']);
            k=k+1;
        end
    end
end
