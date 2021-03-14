input="/Users/Ellen/Documents/GitHub/6862-project/data/Argo_Data/GetArgoData_TextFiles/goodincois.txt"
while IFS= read -r line
do
echo "$line"
rsync -avzh --delete vdmzrs.ifremer.fr::argo/$line /Users/Ellen/Desktop/ArgoGDAC/dac/incois
done < "$input"
