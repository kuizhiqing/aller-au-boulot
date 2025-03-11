# controller

`Scheme`
src/k8s.io/apimachinery/pkg/runtime/scheme.go

* serializing and deserializing API objects
* converting group, version, and kind information
* foundation for a versioned API and versioned configuration


```
SchemeBuilder          = runtime.NewSchemeBuilder(addKnownTypes, addDefaultingFuncs)
AddToScheme            = SchemeBuilder.AddToScheme

var scheme = runtime.NewScheme()
AddToScheme(scheme)
```
