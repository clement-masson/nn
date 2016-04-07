local ParallelCriterion, parent = torch.class('nn.ParallelCriterion', 'nn.Criterion')

function ParallelCriterion:__init(repeatTarget)
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
   self.repeatTarget = repeatTarget
   self.allCosts = {}
end

function ParallelCriterion:add(criterion, weight)
   assert(criterion, 'no criterion provided')
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

function ParallelCriterion:updateOutput(input, target)
   self.output = 0
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
      self.allCosts[i] = self.criterions[i].allCosts or self.criterions[i].output
   end
   return self.output
end

function ParallelCriterion:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
   end
   return self.gradInput
end

function ParallelCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end

function ParallelCriterion:printCosts(indent)
   local indent = indent or 0
   for i,criterion in ipairs(self.criterions) do
      if criterion.printCosts then
         criterion:printCosts(indent+1)
      else
         print(string.rep('  ',indent)..criterion.output)
      end
   end
end    
