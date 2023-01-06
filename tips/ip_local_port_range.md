# ip_local_port_range

## Tests

preset
```bash
echo "61000 61001" | sudo tee /proc/sys/net/ipv4/ip_local_port_range
61000 61001

cat /proc/sys/net/ipv4/ip_local_port_range
61000       61001
```

normally,
```bash
nohup nc 123.125.114.144 80 -v &
[1] 16196
nohup: ignoring input and appending output to 'nohup.out'

nohup nc 123.125.114.144 80 -v &
[2] 16197
nohup: ignoring input and appending output to 'nohup.out'

ss -ant |grep 10.0.2.15:61
ESTAB   0        0                10.0.2.15:61001       123.125.114.144:80
ESTAB   0        0                10.0.2.15:61000       123.125.114.144:80
```

as expected,
```bash
nc 123.125.114.144 80 -v
nc: connect to 123.125.114.144 port 80 (tcp) failed: Cannot assign requested address
```

while
```bash
nohup nc 123.125.114.144 443 -v &
[3] 16215
nohup: ignoring input and appending output to 'nohup.out'

nohup nc 123.125.114.144 443 -v &
[4] 16216
nohup: ignoring input and appending output to 'nohup.out'

ss -ant |grep 10.0.2.15:61
ESTAB   0        0                10.0.2.15:61001       123.125.114.144:443
ESTAB   0        0                10.0.2.15:61001       123.125.114.144:80
ESTAB   0        0                10.0.2.15:61000       123.125.114.144:443
ESTAB   0        0                10.0.2.15:61000       123.125.114.144:80
```

further,
```bash
nohup nc 220.181.57.216 80 -v &
[5] 16222
nohup: ignoring input and appending output to 'nohup.out'

nohup nc 220.181.57.216 80 -v &
[6] 16223
nohup: ignoring input and appending output to 'nohup.out'

nc 220.181.57.216 80 -v
nc: connect to 220.181.57.216 port 80 (tcp) failed: Cannot assign requested address

ss -ant |grep :80
SYN-SENT  0        1               10.0.2.15:61001       220.181.57.216:80
SYN-SENT  0        1               10.0.2.15:61000       220.181.57.216:80
SYN-SENT  0        1               10.0.2.15:61001      123.125.114.144:80
SYN-SENT  0        1               10.0.2.15:61000      123.125.114.144:80
```

Above test are available since linux kernel 3.2.

## Reference

* [test blog](https://mozillazg.com/2019/05/linux-what-net.ipv4.ip_local_port_range-effect-or-mean.html)
* [info](https://marc.info/?l=haproxy&m=139315478227467&w=2)
