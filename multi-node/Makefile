all: deploy

install:
	dsh add tektoncd-pipeline-0.11.0-rc3
	dsh add tektoncd-cli-0.8.0
	dsh add kubectl-stable
	dsh add kustomize-3.5.4

uninstall:
	make clean

deploy:
	kubectl apply -f multi-node.yaml

clean:
	kubectl delete namespace kubeflow
	kubectl delete clusterrolebindings.rbac.authorization.k8s.io mpi-operator
	kubectl delete clusterroles.rbac.authorization.k8s.io mpi-operator
	kubectl delete crd mpijobs.kubeflow.org

status:
	kubectl get deployment -n kubeflow -oname
