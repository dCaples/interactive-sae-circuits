/**
 * frontend/src/components/TaskSelector.tsx
 */

import React from "react";

interface TaskSelectorProps {
  selectedTask: string;
  setSelectedTask: (task: string) => void;
}

export const TaskSelector: React.FC<TaskSelectorProps> = ({ selectedTask, setSelectedTask }) => {
  // Hardcode some possible tasks for the UI:
  const tasks = [
    { value: "dictionaryKeyTask", label: "Dictionary Key" },
    { value: "listIndexTask", label: "List Index" },
    { value: "svaTask", label: "Subject-Verb Agreement" },
  ];

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label>Task: </label>
      <select
        value={selectedTask}
        onChange={(e) => setSelectedTask(e.target.value)}
      >
        {tasks.map((t) => (
          <option key={t.value} value={t.value}>
            {t.label}
          </option>
        ))}
      </select>
    </div>
  );
};
