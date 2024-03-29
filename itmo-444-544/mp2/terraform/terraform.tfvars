# These are your values that would be in arguments.txt
imageid                = "ami-0e36378eee3c18b57"
instance-type          = "t2.micro"
key-name               = "itmo444vagrantkey"
vpc_security_group_ids = "sg-0fafd233da91b97bd"
cnt                    = 3
install-env-file       = "install-env.sh"
elb-name               = "jrh-itmo444-elb"
tg-name                = "jrh-itmo-444-tg"
asg-name               = "jrh-itmo444-asg"
lt-name                = "jrh-itmo-444-lt"
db-name                = "company"
db-name-rpl            = "company-rpl"
iam-profile            = "developer"
sns-topic              = "jrh-sns-topic"
dynamodb-table-name    = "jrh-dynamodb-table"
raw-s3-bucket          = "jrh-raw-bucket"
finished-s3-bucket     = "jrh-finished-bucket"
