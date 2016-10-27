require 'mathn'

class Matrix
  def []=(i, j, value)
    @rows[i][j] = value
  end
end
