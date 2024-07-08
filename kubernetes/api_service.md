# api-service

## Create manifests

```
kubeadm init phase certs all
kubeadm init phase kubeconfig admin
kubeadm init phase etcd local
kubeadm init phase control-plane apiserver
```

## 



## PKI 

```
manifests/etcd.yaml
- --client-cert-auth=true
- --cert-file=/etc/kubernetes/pki/etcd/server.crt
- --key-file=/etc/kubernetes/pki/etcd/server.key
- --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
- --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
- --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
- --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
```


kubeconfig
```
cluster:
  server: https://xxx.xxx
  certificate-authority: pki/ca.crt
user:
  client-certificate:
  client-key:
```

```
manifests/kube-apiserver.yaml
- --client-ca-file=/etc/kubernetes/pki/ca.crt
- --etcd-cafile=/etc/kubernetes/pki/etcd/ca.crt
- --etcd-certfile=/etc/kubernetes/pki/apiserver-etcd-client.crt
- --etcd-keyfile=/etc/kubernetes/pki/apiserver-etcd-client.key
- --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
- --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
- --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
- --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
- --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
- --service-account-key-file=/etc/kubernetes/pki/sa.pub
- --service-account-signing-key-file=/etc/kubernetes/pki/sa.key
- --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
- --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
```

