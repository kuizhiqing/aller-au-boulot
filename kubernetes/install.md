# install kubernetes

## install kubelet kubeadm kubectl

set up yum

```bash
# /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgcheck=0
```

```bash
# /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-aarch64
enabled=1
gpgcheck=0
```

install kubeadm etc.

```bash
yum -y install ipvsadm ipset net-tools jq
yum -y install kubelet-1.23.1 kubeadm-1.23.1 kubectl-1.23.1
```

start kubelet

```bash
systemctl daemon-reload
systemctl enable kubelet
```

check kubeadm、kubelet、kubectl

```bash
kubelet --version
kubeadm version -o yaml
kubectl version --short
```

> Node stop here to join master

---

config image for kubernetes

```
kubeadm config --kubernetes-version=v1.23.1 --image-repository=registry.aliyuncs.com/google_containers images list
```

config kubeadm

```yaml
---
apiVersion: kubeadm.k8s.io/v1beta2
kind: ClusterConfiguration
kubernetesVersion: v1.23.1
clusterName: kubernetes
imageRepository: registry.aliyuncs.com/google_containers
networking:
  serviceSubnet: "10.96.0.0/16"
  podSubnet: "10.100.0.0/16"
  dnsDomain: "cluster.local"
---
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
featureGates:
  SupportIPVSProxyMode: true
mode: ipvs
---
kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
cgroupDriver: systemd
```

deploy kubeadm

```shell
kubeadm init --config=kubeadm-config.yaml --upload-certs | tee /tmp/kubeadm-init.log

egrep 'kubeadm.*join|discovery-token-ca-cert-hash' /tmp/kubeadm-init.log >\$HOME/k8s.add.node.txt
```

kubectl auth

```bash
export KUBECONFIG=/etc/kubernetes/admin.conf
mkdir -p \$HOME/.kube
ln -fs /etc/kubernetes/admin.conf \$HOME/.kube/config
```

enable scheduling on master

```bash
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl taint nodes --all node-role.kubernetes.io/master=:PreferNoSchedule
```

deploy flannel with kube-flannel.yml

```bash
kubectl apply -f kube-flannel.yml
```

## install dashboard (optional)

deploy dashboard

```shell
kubectl apply -f kubernetes-dashboard.yaml
```

kubernetes-dashboard.yaml

## add auth

create sa dashboard-admin

```bash
kubectl create serviceaccount dashboard-admin -n kube-system
kubectl create clusterrolebinding dashboard-admin --clusterrole=cluster-admin --serviceaccount=kube-system:dashboard-admin
```

show secret

```bash
kubectl describe secrets -n kube-system \$(kubectl -n kube-system get secret | awk '/dashboard-admin/{print $1}') | awk '/token:/{print$2}' >$HOME/k8s.token.dashboard.txt
```

```
# dashboard token
cat \$HOME/k8s.token.dashboard.txt
cat \$HOME/k8s.add.node.txt
kubectl get cs
kubectl get nodes
kubectl get pod -A
Local_IP=$(kubectl -n kube-system get cm kubeadm-config -oyaml | awk '/advertiseAddress/{print \$NF}')
echo "  https://${Local_IP}:30000"
```

## reference

#### reset kubeadm

```
kubeadm reset -f
systemctl stop kubelet
ifconfig flannel.1 down
ip link del flannel.1
ifconfig cni0 down
ip link del cni0
ifconfig flannel.1 tunl0
ip link del tunl0
ip link del kube-ipvs0
ip link del dummy0
ip link del tunl0@NONE

ipvsadm --clear
rm -rf /etc/cni/net.d
```

sync time

```
yum -y install ntpdate
ntpdate -u  cn.ntp.org.cn
```

show ip

```
ip route show
```

firewall turn off

```
systemctl stop firewalld
systemctl disable firewalld
```

selinux turn off

```
setenforce 0
sed -i "s/^SELINUX=enforcing/SELINUX=disabled/g" /etc/selinux/config
```

swap turn off

```
swapoff -a
sed -i 's/.*swap.*/#&/' /etc/fstab
```

print join command

```
kubeadm token create --print-join-command
```

kubelet log

```
journalctl -xefu kubelet
```

```
kubectl drain <node-name> --ignore-daemonsets --delete-local-data

kubectl delete node
```

pause node

```
kubectl drain <node-name> --ignore-daemonsets --delete-local-data
kubectl uncordon node <node-name>

kubectl taint nodes node key=value1:NoSchedule
kubectl taint nodes node key:NoSchedule-
```

Restart DNS

```
kubectl -n kube-system rollout restart deployment coredns
```

ifconfig

```
ip link set cni0 down
brctl delbr cni0

ifconfig cni0 down
ifconfig flannel.1 down
ifconfig docker0 down
```

## Ubuntu

```
echo "deb https://mirrors.aliyun.com/kubernetes/apt/ kubernetes-xenial main" >>/etc/apt/sources.list.d/kubernetes.list
```

## upgrade

```
# renew certs
kubeadm init phase certs apiserver --config kubeadm-config.yaml
# upgrade
kubeadm upgrade apply --config kubeadm-config.yaml
# restart kubelet
systemctl restart kubelet.service
# check config
kubeadm config view
# check certSANs
openssl x509 -in /etc/kubernetes/pki/apiserver.crt -noout -text
```

## image gc

```
/var/lib/kubelet/config.yaml

imageGCHighThresholdPercent: 99
imageGCLowThresholdPercent: 98


NodeHasDiskPressure
/etc/default/kubelet (for DEBs), or /etc/sysconfig/kubelet (for RPMs).
vi /etc/sysconfig/kubelet
KUBELET_EXTRA_ARGS=--eviction-hard=nodefs.available<5% --eviction-hard=imagefs.available<3%
```
