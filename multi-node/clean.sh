#!/usr/bin/env bash
kubectl delete -n kubeflow application kubeflow
kubectl delete -n kubeflow application mpi-operator
kubectl delete namespace kubeflow
kubectl delete clusterrolebindings.rbac.authorization.k8s.io mpi-operator
kubectl delete clusterroles.rbac.authorization.k8s.io mpi-operator
kubectl delete crd mpijobs.kubeflow.org
