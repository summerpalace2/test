package com.example.next;


import java.util.List;

public class BirthdaybookData {
    /**
     * reason
     */
    public List<ResultDTO> data;
    private String reason;
    /**
     * result
     */
    private ResultDTO result;
    /**
     * error_code
     */
    private Integer error_code;

    public String getReason() {
        return reason;
    }
    public  List<ResultDTO> getData()
    {
        return data;
    }
    public void setData(List<ResultDTO> data) {
        this.data = data;
    }
    public void addData(ResultDTO data) {
        this.data.add(data);
    }
    public void setReason(String reason) {
        this.reason = reason;
    }

    public ResultDTO getResult() {
        return result;
    }

    public void setResult(ResultDTO result) {
        this.result = result;
    }

    public Integer getError_code() {
        return error_code;
    }

    public void setError_code(Integer error_code) {
        this.error_code = error_code;
    }

    public static class ResultDTO {
   
        /**
         * title
         */
        private String title;
        /**
         * birthday
         */
        private String birthday;
        /**
         * nature
         */
        private String nature;
        /**
         * love
         */
        private String love;
        /**
         * money
         */
        private String money;
        /**
         * business
         */
        private String business;
        /**
         * health
         */
        private String health;
        /**
         * lucky_num
         */
        private String lucky_num;
        /**
         * in_love
         */
        private String in_love;
        /**
         * friend
         */
        private String friend;

        public String getTitle() {
            return title;
        }

        public void setTitle(String title) {
            this.title = title;
        }

        public String getBirthday() {
            return birthday;
        }

        public void setBirthday(String birthday) {
            this.birthday = birthday;
        }

        public String getNature() {
            return nature;
        }

        public void setNature(String nature) {
            this.nature = nature;
        }

        public String getLove() {
            return love;
        }

        public void setLove(String love) {
            this.love = love;
        }

        public String getMoney() {
            return money;
        }

        public void setMoney(String money) {
            this.money = money;
        }

        public String getBusiness() {
            return business;
        }

        public void setBusiness(String business) {
            this.business = business;
        }

        public String getHealth() {
            return health;
        }

        public void setHealth(String health) {
            this.health = health;
        }

        public String getLucky_num() {
            return lucky_num;
        }

        public void setLucky_num(String lucky_num) {
            this.lucky_num = lucky_num;
        }

        public String getIn_love() {
            return in_love;
        }

        public void setIn_love(String in_love) {
            this.in_love = in_love;
        }

        public String getFriend() {
            return friend;
        }

        public void setFriend(String friend) {
            this.friend = friend;
        }
    }
}

