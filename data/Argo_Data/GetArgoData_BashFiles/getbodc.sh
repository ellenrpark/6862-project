input="/Users/Ellen/Documents/Python/StartUp/GetArgoData_TextFiles/goodbodc.txt"
while IFS= read -r line
do
echo "$line"
rsync -avzh --delete vdmzrs.ifremer.fr::argo/$line /Users/Ellen/Documents/Python/StartUp/miniDac/dac/bodc
done < "$input"
