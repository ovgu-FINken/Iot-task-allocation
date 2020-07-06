#pragma once
#include <memory>
#include "Task.h"
#include "TaskFactory.h"

class SendTask : public Task {
    public:
        virtual void Execute() override;
        static std::unique_ptr<Task> CreateMethod();
        static std::string GetFactoryName();
    private:
        static bool s_registered;
};
