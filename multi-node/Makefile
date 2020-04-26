all: deploy

install:
	echo dsh add kubectl-stable kustomize-3.5.4 kfctl-1.0.1 tektoncd-pipeline-0.11.0-rc3 tektoncd-cli-0.8.0

uninstall:
	make clean

build:
	kfctl build -f kfctl-multi-node.yaml

generate:
	kustomize build kustomize/application-crds > multi-node.yaml
	echo '---' >> multi-node.yaml
	kustomize build kustomize/application >> multi-node.yaml
	echo '---' >> multi-node.yaml
	kustomize build kustomize/mpi-operator >> multi-node.yaml

deploy:
	kubectl apply -f application-crds.yaml && \
	kubectl apply -f application.yaml && \
	kubectl apply -f multi-node.yaml

run-job:
	kubectl apply -f tensorflow-horovod-mnist.yaml

clean:
	kubectl delete -n kubeflow application kubeflow
	kubectl delete -n kubeflow application mpi-operator
	kubectl delete namespace kubeflow
	kubectl delete clusterrolebindings.rbac.authorization.k8s.io mpi-operator
	kubectl delete clusterroles.rbac.authorization.k8s.io mpi-operator
	kubectl delete crd mpijobs.kubeflow.org

validate:
	kubectl apply -f multi-node.yaml --dry-run=client --validate

status:
	kubectl logs -n kubeflow $(shell kubectl get -n kubeflow pods -oname | sed 's/^pod\///')
