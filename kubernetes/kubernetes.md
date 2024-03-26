# kubernetes

## Doc

[Document](https://kubernetes.io/docs/home/)


### Design

[GitHub - kubernetes/design-proposals-archive: Archive of Kubernetes Design Proposals](https://github.com/kubernetes/design-proposals-archive)

### socket

```bash
dockerd -H unix:///var/run/docker.sock -H tcp://192.168.59.106 -H tcp://10.10.10.2

curl -XPOST --unix-socket /var/run/docker.sock -d '{"Image":"nginx"}' -H 'Content-Type: application/json' http://localhost/containers/create
{"Id":"1b46f62945c70c0f21e00cb72db968201352e6536072278e23674419c4d763ee","Warnings":[]}

curl -XPOST --unix-socket /var/run/docker.sock http://localhost/containers/1b46f62945c70c0f21e00cb72db968201352e6536072278e23674419c4d763ee/start

curl --unix-socket /var/run/docker.sock http://localhost/events
```


### Linux Network

Linux namespaces

* Cgroups

* IPC

* Network

* Mount

* User

* PID

* UTS(Unix Time Sharing), NIS(network information service)

## Reference

* [veth(4) - Linux manual page](https://man7.org/linux/man-pages/man4/veth.4.html)

* [Linux Network Namespaces](https://matrix.ai/blog/linux-network-namespaces/)
