
maindac='/Users/Ellen/Documents/Python/StartUp/miniDac/dac/'
for dac in $maindac*
do
dacfloat=$dac/*
for float in $dacfloat
do
echo $float
rm $float/*_prof.nc
rm $float/*_meta.nc
rm $float/*_tech.nc
echo Prof, meta, and tech files removed
done
done
