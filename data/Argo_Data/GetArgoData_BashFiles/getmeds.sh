input="/Users/Ellen/Documents/Python/StartUp/GetArgoData_TextFiles/goodmeds.txt"
while IFS= read -r line
do
echo "$line"
rsync -avzh --delete vdmzrs.ifremer.fr::argo/$line /Volumes/Data/Ellen_Data/Science_Data/ArgoGDAC/dac/meds
done < "$input"
