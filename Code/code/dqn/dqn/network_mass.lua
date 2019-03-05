
require 'network'

return function(args)
    args.n_hid1          = 20
    args.n_hid2          = 20
    -- args.nl             = nn.Sigmoid
    args.nl             = nn.Rectifier

    return create_network(args)
end

