require './utils/matrix'
require 'byebug'

# Multilayer Perceptron
class Perceptron
  ALPHA = 1
  BETA = 1
  CLASSES_COUNT = 2
  NEURONS_COUNT = 4
  D = 0.8

  def initialize(benchmarks)
    initialize_benchmarks(benchmarks)
  end

  def train
    initialize_knowledge
    initialize_temp_elements
    loop do
      @benchmarks.each do |b|
        start_one_cycle_of_training(b)
      end
      break if @all_d.flatten.map(&:abs).max < D
      initialize_temp_elements
    end
  end

  def determine_object_class(object)
    clear_layers
    pass_through_g(object)
    pass_through_y
    p @y
    @y.index(@y.max)
  end

  private

  def initialize_benchmarks(benchmarks)
    @benchmarks = benchmarks
  end

  # def initialize_benchmarks(b)
  #   @max = []
  #   @min = []
  #   (0...b.first[:x].count).each do |i|
  #     column = []
  #     b.each do |e|
  #       column.push e[:x][i]
  #     end
  #     @max.push column.max
  #     @min.push column.min
  #     b.each_with_index do |e, j|
  #       b[j][:x][i] = e[:x][i] - @min[i] <= @max[i] - e[:x][i] ? -1 : 1
  #     end
  #   end
  #   @benchmarks = b
  # end

  def start_one_cycle_of_training(benchmark)
    pass_through_g(benchmark[:x])
    pass_through_y
    calculate_d(benchmark)
    calculate_e
    recalculate_brains(benchmark[:x])
  end

  def initialize_knowledge
    @q = Array.new(NEURONS_COUNT / 2) { rand }
    @t = Array.new(CLASSES_COUNT) { rand }
    @v = Matrix.build(NEURONS_COUNT / 2, NEURONS_COUNT) { rand }
    @w = Matrix.build(CLASSES_COUNT, NEURONS_COUNT / 2) { rand }
  end

  def initialize_temp_elements
    @g = Array.new(NEURONS_COUNT / 2) { nil }
    @y = Array.new(CLASSES_COUNT) { nil }
    @d = Array.new(CLASSES_COUNT) { nil }
    @all_d = []
    @e = Array.new(NEURONS_COUNT / 2) { nil }
  end

  def clear_layers
    @g = Array.new(NEURONS_COUNT / 2) { nil }
    @y = Array.new(CLASSES_COUNT) { nil }
  end

  def pass_through_g(benchmark)
    @g.each_with_index do |_e, j|
      sum = 0
      benchmark.each_with_index { |b, i| sum += @v[j, i] * b }
      @g[j] = activation_function(sum + @q[j])
    end
  end

  def pass_through_y
    @y.each_with_index do |_e, k|
      sum = 0
      @g.each_with_index { |g, j| sum += @w[j, k] * g }
      @y[k] = activation_function(sum + @t[k])
    end
  end

  def calculate_d(benchmark)
    d = (Matrix[benchmark[:y]] - Matrix[@y]).to_a.flatten
    @d = d
    @all_d.push d
  end

  def calculate_e
    @e.each_with_index do |_e, j|
      sum = 0
      @y.each_with_index { |y, k| sum += @d[k] * y * (1 - y) * @w[k, j] }
      @e[j] = activation_function(sum)
    end
  end

  def recalculate_brains(benchmark)
    recalculate_w
    recalculate_t
    recalculate_v(benchmark)
    recalculate_q
  end

  def recalculate_w
    @w.each_with_index do |_e, k, j|
      @w[k, j] = @w[k, j] + ALPHA * @y[k] * (1 - @y[k]) * @d[k] * @g[j]
    end
  end

  def recalculate_t
    @t.each_with_index do |_e, k|
      @t[k] = @t[k] + ALPHA * @y[k] * (1 - @y[k]) * @d[k]
    end
  end

  def recalculate_v(benchmark)
    @v.each_with_index do |_e, j, i|
      @v[j, i] = @v[j, i] + BETA * @g[j] * (1 - @g[j]) * @e[j] * benchmark[i]
    end
  end

  def recalculate_q
    @q.each_with_index do |_e, j|
      @q[j] = @q[j] + BETA * @g[j] * (1 - @g[j]) * @e[j]
    end
  end

  def activation_function(x)
    1 / (1 + Math.exp(-x))
  end
end
