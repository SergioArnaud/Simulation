from complete_period import complete_period

a,c,m = 13,13,16
print(complete_period(a,c,m))

a,c,m = 12,13,16
complete_period(a,c,m)

a,c,m = 13,12,16
complete_period(a,c,m)

a,c,m = 1,12,13
complete_period(a,c,m)

a,c,m = 2814749767109, 59482661568307,2**48
complete_period(a,c,m)


