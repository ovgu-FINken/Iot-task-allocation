#include "TaskFactory.h"
#include "SendTask.h"



std::map<std::string, TaskFactory::TCreateMethod> TaskFactory::s_methods;

bool TaskFactory::Register(const std::string name, TCreateMethod funcCreate) {
    if (auto it = s_methods.find(name); it == s_methods.end()){
        s_methods[name] = funcCreate;
        return true;
    }
    return false;
}

std::unique_ptr<Task> TaskFactory::Create(const std::string& name){
    if (auto it = s_methods.find(name); it != s_methods.end()) {
        return it->second();
    }
    return nullptr;
}

