#pragma once

#include <string>
#include <memory>
#include <map>
#include "Task.h"

enum class TaskType {Sensing =1, Relaying = 2, Actuating = 3};

class TaskFactory{
    public:
        using TCreateMethod = std::unique_ptr<Task>(*)();
    public:
        TaskFactory() = delete;
        static bool Register(const std::string name, TCreateMethod funcCreate);
        static std::unique_ptr<Task> Create(const std::string& name);
    private:
        static std::map<std::string, TCreateMethod> s_methods;
};
